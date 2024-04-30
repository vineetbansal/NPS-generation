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
        "--ranks_file",
        type=str,
        help="Path to ranks file ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot_generated_v_ref(outcome, output_dir):
    data = []
    # Fill all the empty ranks with 0
    outcome["target_rank"].fillna(0, inplace=True)
    data.append(outcome[outcome["target_rank"] == 0]["nn_tc"])
    data.append(outcome[~(outcome["target_rank"] == 0)]["nn_tc"])

    labels = ["Ever Generated", "Never Generated"]

    sns.violinplot(
        data=data,
        palette=sns.color_palette("pastel"),
        linecolor="auto",
        width=0.2,
        inner="box",
        inner_kws={"box_width": 3},
        native_scale=True,
    )

    plt.title(
        "nearest-neighbor Tc for held-out molecules that were ever generated vs. never generated"
    )
    plt.ylabel("Nearest-neighbor Tc")
    plt.xticks(list(range(len(labels))), labels)

    file_name = Path(output_dir) / "nn_tc_ever_v_never"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot(outcome_dir, ranks_file, output_dir):
    # Make an output directory if it doesn't yet
    os.makedirs(output_dir, exist_ok=True)

    outcome_files = glob.glob(f"{outcome_dir}/*_nn_tc_ever_v_never.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )
    rank_df = pd.read_csv(ranks_file)
    rank_df = rank_df[rank_df["target_source"] == "model"]

    merged_df = pd.merge(outcome, rank_df, how="inner", on=["smiles"])
    plot_generated_v_ref(merged_df, output_dir)


def main(args):
    plot(
        outcome_dir=args.outcome_dir,
        ranks_file=args.ranks_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
