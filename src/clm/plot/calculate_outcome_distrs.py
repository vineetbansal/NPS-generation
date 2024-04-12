import argparse
import glob
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import defaultdict


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


def plot_continuous(outcomes, output_dir, value_map):
    for outcome, df in outcomes.items():
        data = {
            value_map[source]: value["value"].reset_index(drop=True)
            for source, value in df.groupby("source")
        }
        sns.kdeplot(
            data=data,
            palette=sns.color_palette("pastel"),
        )

        plt.title(outcome)
        plt.ylabel("Density")
        plt.xlabel(outcome)

        file_name = Path(output_dir) / outcome
        plt.savefig(file_name)

        # Clear the previous figure
        plt.clf()


def plot_discrete(split_outcomes, output_dir, value_map, sources):
    results = defaultdict(list)
    category_names = [i.split(" ")[-1] for i in split_outcomes.keys()]

    for outcome, df in split_outcomes.items():
        for source in sources:
            if df[df["source"] == source].shape[0] == 0:
                results[value_map[source]].append(0)
            else:
                results[value_map[source]].append(
                    sum(df[df["source"] == source]["value"].values)
                )

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap("RdYlGn")(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)

        r, g, b, _ = color
    ax.legend(
        ncol=len(category_names),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize="small",
    )

    plt.tight_layout()

    file_name = Path(output_dir) / "Percent of atoms"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot(outcome_dir, output_dir):
    # Make output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    outcome_files = glob.glob(f"{outcome_dir}/*calculate_outcome_distrs.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )

    value_map = {
        "model": "Generated Molecules",
        "train": "Known NPSs",
        "pubchem": "Pubchem",
    }
    # sources = set(list(outcome.source))
    continuous, discrete_atoms, discrete_types = {}, {}, {}
    for outcome, df in outcome.groupby("outcome"):
        if str(outcome).startswith("# atoms"):
            discrete_atoms[outcome] = df
        elif str(outcome).startswith("# of"):
            discrete_types[outcome] = df
        else:
            continuous[outcome] = df
    plot_continuous(continuous, output_dir, value_map)


def main(args):
    plot(outcome_dir=args.outcome_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
