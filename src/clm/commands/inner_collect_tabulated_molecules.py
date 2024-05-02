import argparse
import os
import pandas as pd

from clm.functions import write_to_csv_file

parser = argparse.ArgumentParser()


def add_args(parser):
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Input file paths for sampled molecule data",
    )
    parser.add_argument(
        "--output_file", type=str, help="File path to save the output file"
    )
    return parser


def collect_tabulated_molecules(input_files, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.concat(
        [pd.read_csv(file, sep=",") for file in input_files], ignore_index=True
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

    write_to_csv_file(output_file, unique)


def main(args):
    collect_tabulated_molecules(
        input_files=args.input_files, output_file=args.output_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
