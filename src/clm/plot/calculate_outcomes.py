import argparse
import glob

import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt
import seaborn as sns


def add_args(parser):
    parser.add_argument(
        "--outcome_dir",
        type=str,
        help="Path to directory where all the model evaluation files are saved ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot(outcome_dir, output_dir):

    # Make output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    outcome_files = glob.glob(f"{outcome_dir}/*calculate_outcomes.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )

    # Plot every figure possible
    for outcome_name, _outcome in outcome.groupby("outcome"):
        _data = {}
        for bin, df in _outcome.groupby("bin"):
            _data[bin] = df["value"].tolist()

        # rearrange by bin ascending
        data = []
        labels = []
        sorted_bins = sorted(_data.keys(), key=lambda x: int(x.split("-")[0]))
        for bin in sorted_bins:
            labels.append(bin)
            data.append(_data[bin])

        sns.violinplot(
            data=data,
            palette=sns.color_palette("pastel"),
            linecolor="auto",
            width=0.4,
            inner="box",
            inner_kws={"box_width": 3},
            native_scale=True,
        )

        plt.title(outcome_name)
        plt.ylabel(outcome_name)
        plt.xlabel("Frequency")
        plt.xticks(list(range(len(labels))), labels)

        file_name = Path(output_dir) / outcome_name
        plt.savefig(file_name)

        # Clear the previous figure
        plt.clf()


def main(args):
    plot(
        outcome_dir=args.outcome_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
