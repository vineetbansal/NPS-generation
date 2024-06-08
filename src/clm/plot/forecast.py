import argparse
import pandas as pd
import re
from pathlib import Path
import logging
import os
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import seaborn as sns
from clm.functions import read_csv_file

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--outcome_files",
        type=str,
        nargs="+",
        help="Paths of all the model evaluation files relevant to nn_tc_ever_v_never ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot_roc_curve(outcome, output_dir, fold):
    outcome = outcome[outcome["curve"] == "ROC"]

    plt.figure()
    for mode in ("true", "random"):
        _outcome = outcome[outcome["mode"] == mode]
        tpr, fpr = _outcome["tpr"], _outcome["fpr"]
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{mode} ROC curve (area = {roc_auc:.2})",
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve showing recovery of test set molecules (Fold {fold})")
    plt.legend(loc="lower right")

    filepath = Path(output_dir) / f"forecast_roc_{fold}"
    plt.savefig(filepath)

    # Clear the previous figure
    plt.clf()


def plot_distribution(outcome, output_dir):
    logger.info("Plotting frequency distribution")
    enrichment_factor = list(outcome["EF"].dropna())
    rank = list(outcome["rank"].dropna())

    sns.scatterplot(x=rank, y=enrichment_factor)
    plt.title("Fold enrichment for test-set molecules in model output")
    plt.xlabel("# of top-ranked molecules")
    plt.ylabel("Fold enrichment")

    # Plot logarithmic scale
    plt.yscale("log")
    plt.xscale("log")

    file_path = Path(output_dir) / "fold_enrichment"
    plt.savefig(file_path)

    # Clear the previous figure
    plt.clf()


def plot(outcome_files, output_dir):
    # Make an output directory if it doesn't yet
    os.makedirs(output_dir, exist_ok=True)

    combined_outcome = []
    for outcome_file in outcome_files:
        fold = re.search(r"(\d)", os.path.basename(outcome_file)).group(1)
        outcome = read_csv_file(outcome_file, delimiter=",")
        combined_outcome.append(outcome)
        plot_roc_curve(outcome, output_dir, fold=fold)

    combined_outcome = pd.concat(combined_outcome)
    plot_distribution(combined_outcome, output_dir)


def main(args):
    plot(
        outcome_files=args.outcome_files,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
