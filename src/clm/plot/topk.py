import argparse
import seaborn as sns
from pathlib import Path
import os
from matplotlib import pyplot as plt
from clm.functions import read_csv_file


def add_args(parser):
    parser.add_argument(
        "--rank_file",
        type=str,
        help="Paths of all the model evaluation files relevant to calculate outcomes ",
    )
    parser.add_argument(
        "--tc_file",
        type=str,
        help="Paths of all the model evaluation files relevant to calculate outcomes ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def topk(df, category, output_dir, target_column, title, filename):
    ys = {item: [] for item in category}
    ks = range(0, 30)
    for k in ks:
        for item in category:
            rows = df[df[target_column] == item]
            n_rows = len(rows)  # independent of k, so can be pulled out of loop
            n_rows_at_least_rank_k = len(
                rows[rows["target_rank"] <= k]
            )  # filters out NaNs
            top_k_accuracy = n_rows_at_least_rank_k / n_rows

            ys[item].append(top_k_accuracy)

    for model in category:
        plt.step(ks, ys[model], label=model)
    plt.title(title)
    plt.xlabel("k")
    plt.xscale("log")
    plt.ylabel("Accuracy (%) ")
    plt.legend()

    file_name = Path(output_dir) / filename
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def topk_box(df, output_dir):
    models = ("model", "model (random)", "PubChem", "addcarbon", "train")
    data = []
    for model in models:
        if model == "model (random)":
            outcome = df[df["target_source"] == "model"]
            data.append(
                list(
                    outcome["Tc"].sample(
                        n=200,
                    )
                )
            )
        else:
            outcome = df[df["target_source"] == model]
            data.append(list(outcome[outcome["target_rank"] == 0]["Tc"]))

    sns.boxplot(data=data)
    plt.xticks(list(range(len(models))), models)

    plt.title("Structural prior evaluation plots")
    # plt.xscale("log")
    plt.ylabel("Tanimoto coefficient ")
    plt.legend()

    file_name = Path(output_dir) / "topk_box_plot"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def plot(rank_file, tc_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    rank_df = read_csv_file(rank_file, delimiter=",")
    tc_df = read_csv_file(tc_file, delimiter=",")

    topk(
        rank_df,
        ("model", "PubChem", "addcarbon", "train"),
        output_dir,
        target_column="target_source",
        title="Figure 6b",
        filename="topk",
    )
    topk_box(
        tc_df,
        output_dir,
    )


def main(args):
    plot(
        rank_file=args.rank_file,
        tc_file=args.tc_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
