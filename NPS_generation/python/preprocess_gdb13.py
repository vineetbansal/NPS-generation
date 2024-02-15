"""
Preprocess the entire GDB-13 dataset.
"""

import argparse
import os
from rdkit import Chem


# import functions
from NPS_generation.python.functions import (
    clean_mol,
    remove_salts_solvents,
    NeutraliseCharges,
)


def add_args(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    return parser


def preprocess_gdb13(input_file, output_file):
    # check output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # set up output
    if os.path.exists(output_file):
        os.remove(output_file)

    f_out = open(output_file, "a+")

    # read SMILES
    print("reading input SMILES ...")
    with open(input_file) as f:
        for line_idx, line in enumerate(f):
            # split line
            split = line.strip().split("\t")
            smiles = split[0]

            # convert to molecule
            mol = clean_mol(smiles)
            if mol is None:
                next
            # remove salts/solvents
            mol = remove_salts_solvents(mol, hac=3)
            if mol is None:
                next
            # remove charges
            mol = NeutraliseCharges(mol)
            if mol is None:
                next
            # convert back to canonical SMILES
            canonical = Chem.MolToSmiles(mol)
            if canonical == "":
                next

            # append to file
            _ = f_out.write(canonical + "\n")
            f_out.flush()


def main(args):
    preprocess_gdb13(input_file=args.input_file, output_file=args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
