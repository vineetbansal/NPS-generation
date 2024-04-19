import argparse
import glob
from pathlib import Path
from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os


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


def plot_distribution(novel_outcomes, output_dir):
    data = list(novel_outcomes[True])

    x, y = [], []
    for key, val in Counter(data).items():
        x.append(key)
        y.append(val)

    sns.scatterplot(x=x, y=y)
    plt.title("Frequency Distribution Plot")
    plt.xlabel("Sampling Frequency")
    plt.ylabel("# of unique molecules")

    # Plot logarithmic scale
    plt.yscale("log")
    plt.xscale("log")

    file_path = Path(output_dir) / "freq_distr_scatter"
    plt.savefig(file_path)

    # Clear the previous figure
    plt.clf()


def plot_box_plot(novel_outcomes, output_dir):
    data = [list(novel_outcomes[True]), list(novel_outcomes[False])]
    sns.boxplot(data=data, width=0.2)

    plt.title("Frequency with which test set molecules sampled from a given CV fold")
    plt.ylabel("Sampling Frequency")
    plt.xticks([0, 1], ["Novel Molecules", "Known Molecules"])
    plt.yscale("log")

    file_path = Path(output_dir) / "freq_distr_box"
    plt.savefig(file_path)

    # Clear the previous figure
    plt.clf()


def plot(outcome_dir, output_dir):
    # Make output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    outcome_files = glob.glob(f"{outcome_dir}/*freq_distribution.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )

    # Dictionary of novelty (True or False) mapped to list of frequency of generation
    # E.g. True : [1, 3, 4, 5] means there are four distinct novel smiles
    # each of which were generated  once, thrice, four, and five times
    novel_outcomes = {
        is_novel: df["size"] for is_novel, df in outcome.groupby("is_novel")
    }

    plot_distribution(novel_outcomes, output_dir)
    plot_box_plot(novel_outcomes, output_dir)


def main(args):
    plot(
        outcome_dir=args.outcome_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
