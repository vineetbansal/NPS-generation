import argparse
import pandas as pd
from clm.functions import set_seed, seed_type

parser = argparse.ArgumentParser(description=__doc__)


def add_args(parser):
    parser.add_argument("--sample_file", type=str, help="Path to the sampled file")
    parser.add_argument(
        "--max_molecules",
        type=int,
        help="Max number of sampled smiles to select for a bin (smiles that are not designated to a bin are discarded.)",
        default=None,
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to the save the output file"
    )
    parser.add_argument(
        "--seed", type=seed_type, default=None, nargs="?", help="Random seed."
    )
    return parser


def prep_outcomes_freq(sample_file, max_molecules, output_file, seed=None):
    set_seed(seed)
    data = pd.read_csv(sample_file)

    # TODO: make this process dynamic later
    frequency_ranges = [(1, 1), (2, 2), (3, 10), (11, 30), (31, 100), (101, None)]

    data["bin"] = ""
    for (f_min, f_max) in frequency_ranges:
        if f_max is not None:
            selected_rows = data[data["size"].between(f_min, f_max)]
        else:
            selected_rows = data[data["size"] > f_min]

        if max_molecules is not None and len(selected_rows) > max_molecules:
            selected_rows = selected_rows.sample(n=max_molecules)

        bin_name = f"{f_min}-{f_max}" if f_max is not None else f"{f_min}-"

        data.loc[selected_rows.index, "bin"] = bin_name

    # Save only the rows where we've assigned a bin
    data = data[data["bin"] != ""].reset_index(drop=True)
    data.to_csv(output_file, index=False)

    return data


def main(args):
    prep_outcomes_freq(
        sample_file=args.sample_file,
        max_molecules=args.max_molecules,
        output_file=args.output_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
