import argparse
import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from clm.functions import read_file, read_file_incremental, clean_mol

# suppress rdkit errors
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")


def add_args(parser):
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file path for sampled molecule data",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Input file path for training data",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="SMILES",
        help="Molecular representation format (one of: SMILES/SELFIES)",
    )
    parser.add_argument(
        "--output_file", type=str, help="File path to save the output file"
    )
    return parser


def tabulate_molecules(input_file, train_file, representation, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    train_smiles = set(read_file(train_file))
    sampled_smiles = read_file_incremental(input_file)

    new_smiles = []
    for i, line in enumerate(tqdm(sampled_smiles)):
        *_, smile = line.split(",")

        # input file may have empty value for smile
        if smile.strip() == "":
            continue

        try:
            mol = clean_mol(smile, selfies=representation == "SELFIE")
        except ValueError:
            continue
        else:
            mass = round(Descriptors.ExactMolWt(mol), 6)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            canonical_smile = Chem.MolToSmiles(mol, isomericSmiles=False)

            if canonical_smile not in train_smiles:
                new_smiles.append([canonical_smile, mass, formula])

    freqs = (
        pd.DataFrame(new_smiles, columns=["smiles", "mass", "formula"])
        .groupby(["smiles", "mass", "formula"])
        .size()
        .to_frame("size")
        .sort_values("size", ascending=False)
        .reset_index()
    )

    freqs.to_csv(output_file, index=False)


def main(args):
    tabulate_molecules(
        input_file=args.input_file,
        train_file=args.train_file,
        representation=args.representation,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
