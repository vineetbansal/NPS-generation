import argparse
from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def add_args(parser):
    parser.add_argument("--outcome_file", type=str, help="Path to input file")
    parser.add_argument(
        "--plot_type", type=str, help="The type of plot you wanna visualize"
    )
    return parser


def plot(outcome_file, plot_type):
    outcome = pd.read_csv(outcome_file)
    novel_outcomes = {
        df["is_novel"].iloc[0]: df["size"] for _, df in outcome.groupby("is_novel")
    }

    if plot_type == "A":
        data = list(novel_outcomes[True])

        x, y = [], []
        for key, val in Counter(data).items():
            x.append(key)
            y.append(val)

        sns.scatterplot(x=x, y=y)
        plt.title("Frequency Distribution Plot")
        plt.xlabel("Sampling Frequency")
        plt.ylabel("# of unique molecules")
        plt.yscale("log")
        plt.xscale("log")

        plt.show()
    else:
        data = [list(novel_outcomes[True]), list(novel_outcomes[False])]
        sns.boxplot(data=data, width=0.2)

        plt.title(
            "Frequency with which test set molecules sampled from a given CV fold"
        )
        plt.ylabel("Sampling Frequency")
        plt.xticks([0, 1], ["Novel Molecules", "Known Molecules"])
        plt.yscale("log")
        plt.show()


def main(args):
    plot(outcome_file=args.outcome_file, plot_type=args.plot_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
