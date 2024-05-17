import argparse
import pandas as pd
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from clm.functions import read_file, clean_mol, write_to_csv_file, read_csv_file

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


def save_unfiltered_smiles(output_file, smiles, type):
    filepath, filename = os.path.dirname(output_file), os.path.basename(output_file)
    smiles_freq = [[smile, smiles.count(smile)] for smile in smiles]

    write_to_csv_file(
        filepath + "/" + type + "_" + filename,
        pd.DataFrame(smiles_freq, columns=["smiles", "size"]),
    )


def tabulate_molecules(input_file, train_file, representation, output_file):
    train_data = read_csv_file(train_file)
    # create a dictionary from inchikey to smiles
    train_data = train_data.set_index("inchikey")["smiles"].to_dict()
    sampled_smiles = read_file(input_file, stream=True)

    new_smiles, invalid_smiles, known_smiles = [], [], []
    for i, line in enumerate(tqdm(sampled_smiles)):
        *_, smile = line.split(",")

        # input file may have empty value for smile
        if smile.strip() == "":
            continue

        try:
            mol = clean_mol(smile, selfies=representation == "SELFIE")
        except ValueError:
            invalid_smiles.append(smile)
        else:
            mass = round(Descriptors.ExactMolWt(mol), 6)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            canonical_smile = Chem.MolToSmiles(mol, isomericSmiles=False)
            inchikey = Chem.inchi.MolToInchiKey(mol)

            if inchikey not in train_data:
                new_smiles.append([canonical_smile, mass, formula, inchikey])
            else:
                known_smiles.append(canonical_smile)

    freqs = pd.DataFrame(new_smiles, columns=["smiles", "mass", "formula", "inchikey"])

    # Find unique combinations of inchikey, mass, and formula, and add a
    # `size` column denoting the frequency of occurrence of each combination.
    # For each unique combination, select the first canonical smile.
    unique = freqs.groupby(["inchikey", "mass", "formula"]).first().reset_index()
    unique["size"] = (
        freqs.groupby(["inchikey", "mass", "formula"]).size().reset_index(drop=True)
    )
    unique = unique.sort_values("size", ascending=False).reset_index(drop=True)

    write_to_csv_file(output_file, unique)
    save_unfiltered_smiles(output_file, known_smiles, type="known")
    save_unfiltered_smiles(output_file, invalid_smiles, type="invalid")


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
