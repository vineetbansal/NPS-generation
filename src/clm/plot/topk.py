import argparse
import pandas as pd
from pathlib import Path
import os
from matplotlib import pyplot as plt
from clm.functions import read_csv_file


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to calculate outcomes ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot(outcome_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.concat(
        [read_csv_file(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )

    print(df.columns)
    print(df.shape)

    models = ("model", "PubChem", "addcarbon", "train")
    ys = {model: [] for model in models}

    ks = range(0, 30)
    for k in ks:
        for model in models:
            rows = df[df["target_source"] == model]
            n_rows = len(rows)  # independent of k, so can be pulled out of loop
            n_rows_at_least_rank_k = len(
                rows[rows["target_rank"] <= k]
            )  # filters out NaNs
            top_k_accuracy = n_rows_at_least_rank_k / n_rows

            ys[model].append(top_k_accuracy)

    for model in models:
        plt.step(ks, ys[model], label=model)
    plt.title("Figure 6b")
    plt.xlabel("k")
    plt.xscale("log")
    plt.ylabel("Top k %")
    plt.legend()

    file_name = Path(output_dir) / "topk"
    plt.savefig(file_name)

    # Clear the previous figure
    plt.clf()


def main(args):
    plot(
        outcome_files=args.outcome_files,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
