import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser()


def add_args(parser):
    parser.add_argument("--input_files", type=str, nargs="+")
    parser.add_argument("--output_file", type=str)
    return parser


def collect_tabulated_molecules(input_files, output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    dfs = []
    for input_file in input_files:
        df = pd.read_csv(input_file, sep=",")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df = (
        df.groupby(["smiles", "mass", "formula"])
        .agg(size=("size", "sum"))
        .reset_index()
    )
    df.to_csv(output_file, index=False)


def main(args):
    collect_tabulated_molecules(
        input_files=args.input_files, output_file=args.output_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
