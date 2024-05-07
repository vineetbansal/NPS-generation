"""
Apply the Renz et al. 'AddCarbon' model to the training set.
"""
import argparse
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

# import functions
from clm.functions import clean_mol, write_smiles, set_seed, read_file, read_csv_file


def add_args(parser):
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--seed", type=int)
    return parser


def add_carbon(input_file, output_file, seed=None):
    set_seed(seed)
    # make output directories
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # remove output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # open buffer
    f = open(output_file, "a+")
    # write header
    row = "\t".join(["input_smiles", "mutated_smiles", "mass", "formula", "inchikey"])
    _ = f.write(row + "\n")
    f.flush()

    # read the input SMILES
    smiles = read_file(input_file, smile_only=False)

    smiles_parts = [sm.split(",") for sm in smiles]
    if all([len(part) == 1 for part in smiles_parts]):
        smiles = [part[0] for part in smiles_parts]
        train_mols = [clean_mol(smile, raise_error=False) for smile in smiles]
        train_inchi = set([Chem.inchi.MolToInchiKey(mol) for mol in train_mols if mol])
    elif all([len(part) == 2 for part in smiles_parts]):
        smiles = [part[0] for part in smiles_parts]
        train_inchi = set([part[1] for part in smiles_parts])
    else:
        raise RuntimeError(
            "The input file should have either 1 column (smile) or 2 columns (smiles, inchikey"
        )

    # loop over the input SMILES
    # output_smiles = list()
    for sm_idx, input_smiles in enumerate(tqdm(smiles)):
        print(
            "working on SMILES {} of {}: '{}' ...".format(
                sm_idx, len(smiles), input_smiles
            )
        )
        """
        code adapted from:
        https://github.com/ml-jku/mgenerators-failure-modes/blob/master/addcarbon.py
        """
        # try all positions in the molecule in random order
        for i in np.random.permutation(len(input_smiles)):
            # insert C at a random spot and check if valid
            mut = input_smiles[:i] + "C" + input_smiles[i:]
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
            # round to 6 decimal places
            mass = round(exact_mass, 6)

            # calculate molecular formula
            formula = rdMolDescriptors.CalcMolFormula(mut_mol)

            # append to file
            row = "\t".join([input_smiles, mut_can, str(mass), formula, mut_inchi])
            _ = f.write(row + "\n")
            f.flush()

        # see if we can break
        # if len(output_smiles) > args.max_smiles:
        #     break

    # write unique SMILES
    uniq_smiles = read_csv_file(output_file, delimiter="\t").mutated_smiles.unique()
    filename, ext = os.path.splitext(output_file)
    uniq_file = filename + "-unique.smi"
    write_smiles(uniq_smiles, uniq_file)


def main(args):
    add_carbon(input_file=args.input_file, output_file=args.output_file, seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
