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

    freq_map = {
        "1-1": "1",
        "2-2": "2",
        "3-10": "3-10",
        "11-30": "11-30",
        "31-100": "31-100",
        "101-": ">100",
    }
    plot_types = {
        "% novel",
        "% unique",
        "% valid",
        "External diversity",
        "External nearest-neighbor Tc",
        "Frechet ChemNet distance",
        "Internal diversity",
        "Internal nearest-neighbor Tc",
        "Jensen-Shannon distance, # of aliphatic rings",
        "Jensen-Shannon distance, # of aromatic rings",
        "Jensen-Shannon distance, # of rings",
        "Jensen-Shannon distance, % rotatable bonds",
        "Jensen-Shannon distance, % sp3 carbons",
        "Jensen-Shannon distance, % stereocenters",
        "Jensen-Shannon distance, Bertz TC",
        "Jensen-Shannon distance, MWs",
        "Jensen-Shannon distance, Murcko scaffolds",
        "Jensen-Shannon distance, NP score",
        "Jensen-Shannon distance, QED",
        "Jensen-Shannon distance, SA score",
        "Jensen-Shannon distance, TPSA",
        "Jensen-Shannon distance, atoms",
        "Jensen-Shannon distance, hydrogen acceptors",
        "Jensen-Shannon distance, hydrogen donors",
        "Jensen-Shannon distance, logP",
        "KL divergence, atoms",
        "Wasserstein distance, atoms",
        "n_mols",
    }

    outcome_files = glob.glob(f"{outcome_dir}/*calculate_outcomes.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )

    # Split the data frame by outcomes and extract the one chosen to be plotted
    split_outcomes = {df["outcome"].iloc[0]: df for _, df in outcome.groupby("outcome")}

    # Plot every figure possible
    for plot_type in plot_types:
        chosen_outcome = split_outcomes[plot_type]

        # Generate a dictionary of frequency bin and respective value to be plotted
        split_freq = {
            freq_map[df["bin"].iloc[0]]: list(df["value"])
            for df in [
                chosen_outcome[chosen_outcome["bin"] == i] for i in freq_map.keys()
            ]
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

        plt.title(plot_type)
        plt.ylabel(plot_type)
        plt.xlabel("Frequency")
        plt.xticks(list(range(len(labels))), labels)

        file_name = Path(output_dir) / plot_type
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
