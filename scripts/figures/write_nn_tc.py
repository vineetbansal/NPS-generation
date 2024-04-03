import argparse
import glob
import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt
import seaborn as sns
from clm.commands.inner_prep_outcomes_freq import prep_outcomes_freq


def add_args(parser):
    parser.add_argument(
        "--outcome_dir",
        type=str,
        help="Path to directory where all the model evaluation files are saved ",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        help="The figure to be plotted: \n"
        "1. 'A' for nn_tc of Generated vs. PubChem \n"
        "2. 'B' for nn_tc by Frequency \n",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot(outcome_dir, plot_type, output_dir):
    # Make an output directory if it doesn't yet
    os.makedirs(output_dir, exist_ok=True)

    source_map = {
        "DeepMet": "Generated metabolites",
        "PubChem": "Negative control (PubChem)",
    }
    outcome_files = glob.glob(f"{outcome_dir}/*write_nn_tc.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )

    if plot_type == "A":
        # Generate a dictionary of source and respective values to be plotted
        split_outcomes = {
            source_map[df["source"].iloc[0]]: list(df["nn_tc"])
            for _, df in outcome.groupby("source")
        }
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

        file_name = Path(output_dir) / "nn_tc_gen_v_pubchem"
        plt.savefig(file_name)
        plt.show()

    else:
        freq_map = {
            "1-1": "1",
            "2-2": "2",
            "3-10": "3-10",
            "11-30": "11-30",
            "31-100": "31-100",
            "101-": ">100",
        }

        # Split the outcomes by frequency bins
        outcome_freq = prep_outcomes_freq(
            outcome, max_molecules=10_000_000, output_file=None
        )

        # Generate a dictionary of frequency bin and respective value (nn_tc) to be plotted
        split_freq = {
            freq_map[df["bin"].iloc[0]]: list(df["nn_tc"])
            for df in [outcome_freq[outcome_freq["bin"] == i] for i in freq_map.keys()]
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

        plt.title("nearest-neighbor Tc, by frequency")
        plt.ylabel("Nearest-neighbor Tc")
        plt.xlabel("Frequency")
        plt.xticks(list(range(len(labels))), labels)

        file_name = Path(output_dir) / "nn_tc_by_freq"
        plt.savefig(file_name)
        plt.show()


def main(args):
    plot(
        outcome_dir=args.outcome_dir,
        plot_type=args.plot_type,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
