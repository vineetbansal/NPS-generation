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
    outcome_map = {
        "Molecular weight",
        "QED",
        "% sp3 carbons"
    }

    outcome_files = glob.glob(f"{outcome_dir}/*calculate_outcome_distrs.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )

    split_outcomes = {outcome: df for outcome, df in outcome.groupby("outcome")}

    for outcome_type in outcome_map:
        chosen_outcome = split_outcomes[outcome_type]
        split_source = pd.DataFrame({
            source: df['value'].reset_index(drop=True) for source, df in chosen_outcome.groupby('source')
        })

        sns.kdeplot(
            data=split_source,
            palette=sns.color_palette("pastel"),
        )

        plt.title(outcome_type)
        plt.ylabel("Density")
        plt.xlabel(outcome_type)
        file_path = f"scripts/figures/ped_fig/{outcome_type}.png"
        plt.savefig(file_path)
        plt.show()


def main(args):
    plot(outcome_dir=args.outcome_dir, outcome_type=args.outcome_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
