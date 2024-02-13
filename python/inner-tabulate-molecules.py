"""
For a large set of molecules sampled from a model, tabulate unique canonical
SMILES and record the following properties:
- SMILES
- exact mass
- molecular formula
- sampling frequency
"""

import argparse
import os
import pandas as pd
import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# import functions
from functions import clean_mol, read_smiles

# suppress rdkit errors
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
def add_args(parser):
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--representation', type=str)
    parser.add_argument('--output_file', type=str)
    
    return parser


def tabulate_molecules(input_file, train_file, representation, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # read training SMILES
    train_smiles = read_smiles(train_file)

    # set up temporary output
    filename, ext = os.path.splitext(output_file)
    tmp_file = filename + '.temp'
    ## remove file if it exists
    if os.path.exists(tmp_file):
      os.remove(tmp_file)

    f2 = open(tmp_file, 'a+')

    # read SMILES line-by-line, and calculate properties for real molecules
    with open(input_file, 'r') as f1:
        f1_lines = list(f1.readlines())
        for line in tqdm(f1_lines, total=len(f1_lines)):
            # split line
            split = line.strip().split(',')
            if len(split) == 2:
                mass, smiles = split[0], split[1]
            else:
                smiles = split[0]

            # try to parse the molecule
            try:
                mol = clean_mol(smiles, representation=representation)

                # calculate exact mass
                exact_mass = Descriptors.ExactMolWt(mol)
                ## round to 6 decimal places
                mass = round(exact_mass, 6)

                # calculate molecular formula
                formula = rdMolDescriptors.CalcMolFormula(mol)

                # roundtrip to get canonical smiles
                canonical_smile = Chem.MolToSmiles(mol, isomericSmiles=False)

                # optionally, skip
                if not canonical_smile in train_smiles:
                    # append to file
                    row = "\t".join([canonical_smile, str(mass), formula])
                    _ = f2.write(row + '\n')
                    f2.flush()
            except ValueError:
                pass

    # read temporary output, and tabulate frequencies
    # NOTE: group by canonical SMILES and pick the best log-likelihood
    df = pd.read_csv(tmp_file, sep='\t', header=None,
                     names=['smiles', 'mass', 'formula'])
    # calculate frequency of each canonical SMILES
    freqs = df.groupby(['smiles', 'mass', 'formula']).size().to_frame('size').\
        reset_index().sort_values('size', ascending=False)
    # write
    freqs.to_csv(output_file, index=False)

    # remove the tmp_file
    os.remove(tmp_file)


def main(args):
    tabulate_molecules(
        input_file=args.input_file,
        train_file=args.train_file,
        representation=args.representation,
        output_file=args.output_file
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)