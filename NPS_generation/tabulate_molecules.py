"""
For a large set of molecules sampled from a model, tabulate unique canonical
SMILES and record the following properties:
- SMILES
- sampling frequency
- negative log-likelihood
- exact mass
- molecular formula
"""

import argparse
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# import functions
from NPS_generation.functions import clean_mol

# suppress rdkit errors
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")


def main(args_list=None):
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument(
        "--input_files", type=str, nargs="*", help="file(s) containing sampled SMILES"
    )
    args = parser.parse_args(args_list)
    print(args)

    output_dir = os.path.dirname(args.output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # set up temporary output
    filename, ext = os.path.splitext(args.output_file)
    tmp_file = filename + ".temp"
    f2 = open(tmp_file, "a+")

    # read SMILES line-by-line, and calculate properties for real molecules
    for idx, input_file in enumerate(args.input_files):
        print(
            "processing input_file {} of {}: {}".format(
                idx + 1, len(args.input_files), input_file
            )
        )

        # read line-by-line
        with open(input_file) as f1:
            for line_idx, line in enumerate(f1):
                # split line into SMILES/NLL
                split = line.strip().split("\t")
                if len(split) == 2:
                    smiles, nll = split[0], split[1]

                    # try to parse the molecule
                    try:
                        mol = clean_mol(smiles)

                        # calculate exact mass
                        exact_mass = Descriptors.ExactMolWt(mol)
                        # round to 6 decimal places
                        mass = round(exact_mass, 6)

                        # calculate molecular formula
                        formula = rdMolDescriptors.CalcMolFormula(mol)

                        # roundtrip to get canonical smiles
                        canonical_smiles = Chem.MolToSmiles(mol)

                        # append to file
                        row = "\t".join([canonical_smiles, nll, str(mass), formula])
                        _ = f2.write(row + "\n")
                        f2.flush()
                    except ValueError:
                        next

    # read temporary output, and tabulate frequencies
    # NOTE: group by canonical SMILES and pick the best log-likelihood
    df = pd.read_csv(
        tmp_file, sep="\t", header=None, names=["smiles", "nll", "mass", "formula"]
    )
    # first, calculate frequency of each canonical SMILES
    freqs = (
        df.groupby(["smiles", "mass", "formula"]).size().to_frame("size").reset_index()
    )
    # second, extract best NLL per canonical SMILES
    nlls = df.sort_values("nll").groupby("smiles").head(1)
    # combine and write
    result = pd.merge(nlls, freqs)
    result.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
