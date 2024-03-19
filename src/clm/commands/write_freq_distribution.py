import argparse

import pandas as pd
import numpy as np
from clm.functions import read_file


def add_args(parser):
    parser.add_argument("--sampled_file", type=str, help="Path to the sampled file")
    parser.add_argument("--test_file", type=str, help="Path to the test file ")
    parser.add_argument(
        "--output_file", type=str, help="Path to the save the output file"
    )

    return parser


def write_freq_distribution(sampled_file, test_file, output_file):
    sampled_data = pd.read_csv(sampled_file)
    test_smiles = set(read_file(test_file, stream=True, smile_only=True))

    # Label smiles not found in test set as novel
    sampled_data = sampled_data.assign(
        is_novel=np.select(
            [
                sampled_data["smiles"].isin(test_smiles).values,
                ~sampled_data["smiles"].isin(test_smiles).values,
            ],
            [False, True],
        )
    )
    sampled_data["is_novel"] = sampled_data["is_novel"].astype(bool)

    sampled_data.to_csv(output_file, index=False)

    smile_distribution = sampled_data.reset_index(drop=True)
    return smile_distribution


def main(args):
    write_freq_distribution(
        sampled_file=args.sampled_file,
        test_file=args.test_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)