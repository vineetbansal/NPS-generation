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
        "--outcome_type",
        type=str,
        help="The metrics that you want to plot (e.g. 'Jensen-Shannon distance, Murcko scaffolds', 'Frechet ChemNet distance'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot(outcome_dir, outcome_type, output_dir):

    # Make output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    freq_map = {
        "1-1": "1",
        "2-2": "2",
        "3-10": "3-10",
        "11-30": "11-30",
        "31-100": "31-100",
        "101-": ">100",
    }

    outcome_files = glob.glob(f"{outcome_dir}/*calculate_outcomes.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )

    # Split the data frame by outcomes and extract the one chosen to be plotted
    split_outcomes = {df["outcome"].iloc[0]: df for _, df in outcome.groupby("outcome")}
    chosen_outcome = split_outcomes[outcome_type]

    # Generate a dictionary of frequency bin and respective value to be plotted
    split_freq = {
        freq_map[df["bin"].iloc[0]]: list(df["value"])
        for df in [chosen_outcome[chosen_outcome["bin"] == i] for i in freq_map.keys()]
    }

    data = list(split_freq.values())
    labels = list(split_freq.keys())

    sns.violinplot(
        data=data,
        palette=sns.color_palette("pastel"),
        linecolor="auto",
        width=0.4,
        inner="box",
        inner_kws={"box_width": 3},
        native_scale=True,
    )

    plt.title(outcome_type)
    plt.ylabel(outcome_type)
    plt.xlabel("Frequency")
    plt.xticks(list(range(len(labels))), labels)

    file_name = Path(output_dir) / outcome_type
    plt.savefig(file_name)
    plt.show()


def main(args):
    plot(
        outcome_dir=args.outcome_dir,
        outcome_type=args.outcome_type,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
