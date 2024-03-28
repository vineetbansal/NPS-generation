import argparse

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def add_args(parser):
    parser.add_argument("--outcome_file", type=str, help="Path to the write nn tc file")
    parser.add_argument(
        "--plot_type", type=str, help="The metrics that you want to plot"
    )
    return parser


def plot(outcome_file, plot_type):
    source_map = {
        "DeepMet": "Generated metabolites",
        "PubChem": "Negative control (PubChem)",
    }
    outcome = pd.read_csv(outcome_file)

    split_outcomes = {
        source_map[df["source"].iloc[0]]: list(df["nn_tc"])
        for _, df in outcome.groupby("source")
    }

    if plot_type == "A":
        data = list(split_outcomes.values())
        labels = list(split_outcomes.keys())

        sns.violinplot(
            data=data,
            palette=sns.color_palette("pastel"),
            linecolor="auto",
            width=0.2,
            inner="box",
            inner_kws={"box_width": 3},
            native_scale=True,
        )

        plt.title(plot_type)
        plt.ylabel("Nearest-neighbor Tc")
        plt.xticks(list(range(len(labels))), labels)
        plt.show()


def main(args):
    plot(outcome_file=args.outcome_file, plot_type=args.plot_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
