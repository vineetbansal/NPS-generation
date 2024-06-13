import argparse
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import os
from clm.functions import read_csv_file

parser = argparse.ArgumentParser(description=__doc__)


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to topk_tc ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the save the output file",
    )

    return parser

def compute_topk_tc(outcome, min_tcs):
    tc_count = {min_tc: [] for min_tc in min_tcs}
    ks = {min_tc: [] for min_tc in min_tcs}
    n_total = len(outcome)

    for min_tc in min_tcs:
        rows = outcome[outcome["Tc"] >= min_tc]
        for k in range(0, 30):
            n_rows_at_least_rank_k = len(rows[rows["target_rank"] <= k])
            top_k_accuracy = (n_rows_at_least_rank_k / n_total) * 100
            ks[min_tc].append(k)
            tc_count[min_tc].append(top_k_accuracy)

    return ks, min_tcs
def plot(outcome_files, output_dir):
    # Make output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    outcome = pd.concat(
        [read_csv_file(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )
    outcome = outcome[outcome["target_source"] == "model"]

    min_tcs = [0.4, 0.675, 1]

    ks, tc_count = compute_topk_tc(outcome, min_tcs)

    for min_tc in min_tcs:
        plt.step(ks[min_tc], tc_count[min_tc], label=min_tc)

    plt.title("Top-k accuracy curve when considering Tc >= 0.4, 0.675 as 'correct'")
    plt.xlabel("k")
    plt.xscale("log")
    plt.ylabel("Top k %")
    plt.legend()

    file_name = Path(output_dir) / "topk_tc"
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
