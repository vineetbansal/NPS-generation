import argparse
import os
import pandas as pd

from clm.functions import write_to_csv_file, read_csv_file

parser = argparse.ArgumentParser()


def add_args(parser):
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Input file paths for sampled molecule data",
    )
    parser.add_argument(
        "--known_smiles",
        type=str,
        nargs="+",
        help="Input file paths for known sampled molecule data",
    )
    parser.add_argument(
        "--invalid_smiles",
        type=str,
        nargs="+",
        help="Input file paths for invalid sampled molecule data",
    )
    parser.add_argument(
        "--output_file", type=str, help="File path to save the output file"
    )
    return parser


def collect_tabulated_molecules(input_files, output_file, known_smiles, invalid_smiles):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.concat(
        [read_csv_file(file, delimiter=",") for file in input_files], ignore_index=True
    )
    invalid_df = pd.concat(
        [read_csv_file(file, delimiter=",") for file in invalid_smiles],
        ignore_index=True,
    )
    known_df = pd.concat(
        [read_csv_file(file, delimiter=",") for file in known_smiles], ignore_index=True
    )
    # Find unique combinations of inchikey, mass, and formula, and add a
    # `size` column denoting the frequency of occurrence of each combination.
    # For each unique combination, select the first canonical smile.
    unique = df.groupby(["inchikey", "mass", "formula"]).first().reset_index()
    unique["size"] = (
        df.groupby(["inchikey", "mass", "formula"])
        .agg({"size": "sum"})
        .reset_index(drop=True)
    )

    unique_invalid = invalid_df.groupby(["smiles"]).first().reset_index()
    unique_invalid["size"] = (
        invalid_df.groupby(["smiles"]).agg({"size": "sum"}).reset_index(drop=True)
    )

    unique_known = known_df.groupby(["smiles"]).first().reset_index()
    unique_known["size"] = (
        known_df.groupby(["smiles"]).agg({"size": "sum"}).reset_index(drop=True)
    )

    write_to_csv_file(output_file, unique)
    write_to_csv_file(
        os.path.join(
            os.path.dirname(output_file), "invalid_" + os.path.basename(output_file)
        ),
        unique_invalid,
    )
    write_to_csv_file(
        os.path.join(
            os.path.dirname(output_file), "known_" + os.path.basename(output_file)
        ),
        unique_known,
    )


def main(args):
    collect_tabulated_molecules(
        input_files=args.input_files,
        output_file=args.output_file,
        known_smiles=args.known_smiles,
        invalid_smiles=args.invalid_smiles,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
