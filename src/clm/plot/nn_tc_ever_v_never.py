import argparse
import re
import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt
import seaborn as sns
from clm.functions import read_csv_file


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to nn_tc_ever_v_never ",
    )
    parser.add_argument(
        "--rank_files",
        type=str,
        nargs="+",
        help="Path to individual CV ranks file ",
    )
    parser.add_argument(
        "--ranks_file",
        type=str,
        help="Path to overall rank file ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot_generated_v_never(outcome, output_dir):
    data = []

    data.append(list(outcome[outcome["target_rank"].notnull()]["nn_tc"]))
    data.append(list(outcome[outcome["target_rank"].isnull()]["nn_tc"]))

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
        "nearest-neighbor Tc for held-out molecules \n (ever generated vs. never generated)"
    )
    plt.ylabel("Nearest-neighbor Tc")
    plt.xticks(list(range(len(labels))), labels)

    file_name = Path(output_dir) / "nn_tc_ever_v_never"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot_generated_ratio(ranks_file, output_dir):
    # Overall rank file
    rank_df = read_csv_file(ranks_file)
    data = {
        "Ever Generated": len(
            rank_df[rank_df["target_source"] == "model"][
                rank_df["target_rank"].notnull()
            ]
        ),
        "Never Generated": len(
            rank_df[rank_df["target_source"] == "model"][
                rank_df["target_rank"].isnull()
            ]
        ),
    }

    sns.set_style("darkgrid")
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

    plt.pie(list(data.values()), labels=list(data.keys()), colors=colors)
    percentage_generated = (
        data["Ever Generated"]
        / (data["Ever Generated"] + data["Never Generated"])
        * 100
    )
    plt.title(f"{percentage_generated:.2f}% of held-out metabolites generated")

    file_name = Path(output_dir) / "ratio_ever_v_never"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot(outcome_files, rank_files, ranks_file, output_dir):
    # Make an output directory if it doesn't yet
    os.makedirs(output_dir, exist_ok=True)

    merged_df = []
    for outcome_file in outcome_files:
        # Extract the fold of the outcome file
        outcome_fold = re.search(r"(\d+)", os.path.basename(outcome_file)).group(1)
        outcome_df = read_csv_file(outcome_file)

        # Concatenate rank files of all the folds except that of the current outcome file
        rank_df = pd.concat(
            [
                read_csv_file(rank_file)
                for rank_file in rank_files
                if not re.match(rf".*{outcome_fold}.*", os.path.basename(rank_file))
            ]
        )
        rank_df = rank_df[rank_df["target_source"] == "model"]

        merged_df.append(
            pd.merge(
                outcome_df,
                rank_df,
                how="inner",
                left_on="inchikey",
                right_on="target_inchikey",
            )
        )

    plot_generated_v_never(pd.concat(merged_df), output_dir)
    plot_generated_ratio(ranks_file, output_dir)


def main(args):
    plot(
        outcome_files=args.outcome_files,
        rank_files=args.rank_files,
        ranks_file=args.ranks_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
