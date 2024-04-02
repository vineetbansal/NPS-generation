import argparse
import glob

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def add_args(parser):
    parser.add_argument("--outcome_dir", type=str, help="Path to outcome directory")
    parser.add_argument(
        "--outcome_type", type=str, help="The metrics that you want to plot"
    )
    return parser


def plot(outcome_dir, outcome_type):
    # outcomes_map = {'Novel':' % novel',
    #                 'Unique': '% unique',
    #                 'Valid':' % valid',
    #                 'External Diversity': 'External diversity',
    #                 'External nn tc': 'External nearest-neighbor Tc',
    #                 'Frechet ChemNet distance', 'Internal diversity',
    #                 'Internal nearest-neighbor Tc',
    #                 'Jensen-Shannon distance, # of aliphatic rings',
    #                 'Jensen-Shannon distance, # of aromatic rings',
    #                 'Jensen-Shannon distance, # of rings',
    #                 'Jensen-Shannon distance, % rotatable bonds',
    #                 'Jensen-Shannon distance, % sp3 carbons',
    #                 'Jensen-Shannon distance, % stereocenters',
    #                 'Jensen-Shannon distance, Bertz TC',
    #                 'Jensen-Shannon distance, MWs',
    #                 'Jensen-Shannon distance, Murcko scaffolds',
    #                 'Jensen-Shannon distance, NP score',
    #                 'Jensen-Shannon distance, QED',
    #                 'Jensen-Shannon distance, SA score',
    #                 'Jensen-Shannon distance, TPSA', 'Jensen-Shannon distance, atoms', 'Jensen-Shannon distance, hydrogen acceptors', 'Jensen-Shannon distance, hydrogen donors', 'Jensen-Shannon distance, logP', 'KL divergence, atoms', 'Wasserstein distance, atoms'}
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

    split_outcomes = {df["outcome"].iloc[0]: df for _, df in outcome.groupby("outcome")}

    chosen_outcome = split_outcomes[outcome_type]
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
    file_path = f"scripts/figures/ped_fig/{outcome_type}.png"
    plt.savefig(file_path)
    plt.show()


def main(args):
    plot(outcome_dir=args.outcome_dir, outcome_type=args.outcome_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
