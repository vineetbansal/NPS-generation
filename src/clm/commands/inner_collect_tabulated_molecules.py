import argparse
import os
import pandas as pd

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
    df = df.groupby(["smiles", "mass", "formula"], as_index=False).agg({"size": "sum"})
    df.to_csv(output_file, index=False)


def main(args):
    collect_tabulated_molecules(
        input_files=args.input_files, output_file=args.output_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
