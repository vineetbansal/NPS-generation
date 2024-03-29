import argparse
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)


def add_args(parser):
    parser.add_argument("--sample_file", type=str, help="Path to the sampled file")
    parser.add_argument(
        "--max_molecules",
        type=int,
        help="Number of samples to select",
        default=10_000_000,
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to the save the output file"
    )
    return parser


def prep_outcomes_freq(sample_file, max_molecules, output_file):
    data = pd.read_csv(sample_file)

    # TODO: make this process dynamic later
    frequency_ranges = [(1, 1), (2, 2), (3, 10), (11, 30), (31, 100), (101, None)]

    data["bin"] = ""
    for (f_min, f_max) in frequency_ranges:
        if f_max is not None:
            selected_rows = data[data["size"].between(f_min, f_max)]
        else:
            selected_rows = data[data["size"] > f_min]

        if len(selected_rows) > max_molecules:
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
