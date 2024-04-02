"""
Apply the Renz et al. 'AddCarbon' model to the training set.
"""
import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

# import functions
from clm.functions import clean_mol, clean_mols, write_smiles, set_seed

def add_args(parser):
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--seed', type=int)
    return parser


def add_carbon(input_file, output_file, seed=None):
    set_seed(seed)
    # make output directories
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    ## remove output file if it exists
    if os.path.exists(output_file):
      os.remove(output_file)

    # open buffer
    f = open(output_file, 'a+')
    # write header
    row = "\t".join(['input_smiles', 'mutated_smiles', 'mass', 'formula'])
    _ = f.write(row + '\n')
    f.flush()

    # read the input SMILES
    df = pd.read_csv(input_file)
    # extract SMILES
    if 'canonical_smiles' in list(df):
        df = df.dropna(subset=['canonical_smiles'])
        smiles = df['canonical_smiles'].values
    else:
        df = df.dropna(subset=['smiles'])
        smiles = df['smiles'].values

    # calculate inchikeys
    train_mols = clean_mols(smiles)
    train_inchi = [Chem.inchi.MolToInchiKey(mol) for mol in train_mols if mol]

    # loop over the input SMILES
    output_smiles = list()
    for sm_idx, input_smiles in enumerate(tqdm(smiles)):
        print("working on SMILES {} of {}: '{}' ...".format(sm_idx, len(smiles), \
                                                            input_smiles))
        """
        code adapted from:
        https://github.com/ml-jku/mgenerators-failure-modes/blob/master/addcarbon.py
        """
        # try all positions in the molecule in random order
        for i in np.random.permutation(len(input_smiles)):
            # insert C at a random spot and check if valid
            mut = input_smiles[:i] + 'C' + input_smiles[i:]
            try:
                mut_mol = clean_mol(mut)
            except Exception:
                continue
            # catch #2
            if mut_mol is None:
                continue

            # if it is valid, compute canonical smiles
            mut_can = Chem.MolToSmiles(mut_mol, isomericSmiles=False)
            mut_inchi = Chem.inchi.MolToInchiKey(mut_mol)
            # can't be in the training set
            if mut_inchi in train_inchi:
               continue

            # calculate exact mass
            exact_mass = Descriptors.ExactMolWt(mut_mol)
            ## round to 6 decimal places
            mass = round(exact_mass, 6)

            # calculate molecular formula
            formula = rdMolDescriptors.CalcMolFormula(mut_mol)

            # append to file
            row = "\t".join([input_smiles, mut_can, str(mass), formula])
            _ = f.write(row + '\n')
            f.flush()

        # see if we can break
        # if len(output_smiles) > args.max_smiles:
        #     break

    # write unique SMILES
    uniq_smiles = pd.read_csv(output_file, sep='\t').mutated_smiles.\
        unique()
    filename, ext = os.path.splitext(output_file)
    uniq_file = filename + '-unique.smi'
    write_smiles(uniq_smiles, uniq_file)



def main(args):
    add_carbon(input_file=args.input_file, output_file=args.output_file,
               seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)