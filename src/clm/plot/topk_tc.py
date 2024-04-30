import argparse
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import glob
import os

parser = argparse.ArgumentParser(description=__doc__)


def add_args(parser):
    parser.add_argument(
        "--outcome_dir", type=str, required=True, help="Path to the sampled file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the save the output file",
    )

    return parser


def get_tc_range(sample_tc, min_tcs):
    tc_match = 0

    for min_tc in min_tcs:
        if sample_tc >= min_tc:
            tc_match = min_tc
        else:
            break
    return tc_match


def topk_tc(outcome_dir, output_dir):
    # Make output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    outcome_files = glob.glob(f"{outcome_dir}/*CV_tc.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )
    outcome = outcome[outcome["target_source"] == "model"]
    outcome = outcome.dropna(subset=["Tc"])

    min_tcs = [0.4, 0.675, 1]

    tc = outcome["Tc"].apply(lambda x: get_tc_range(x, min_tcs))
    outcome = outcome.assign(min_tc=tc)

    tc_count = {min_tc: [] for min_tc in min_tcs}
    ks = range(0, 30)
    for k in ks:
        for min_tc in min_tcs:
            rows = outcome[outcome["min_tc"] == min_tc]
            n_rows = len(rows)
            n_rows_at_least_rank_k = len(rows[rows["target_rank"] <= k])
            top_k_accuracy = (n_rows_at_least_rank_k / n_rows) * 100
            tc_count[min_tc].append(top_k_accuracy)

    for min_tc in min_tcs:
        plt.step(ks, tc_count[min_tc], label=min_tc)
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
    topk_tc(
        outcome_dir=args.outcome_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
