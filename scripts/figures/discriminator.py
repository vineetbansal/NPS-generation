import argparse
import glob
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


def add_args(parser):
    parser.add_argument("--outcome_dir", type=str, help="Path to input file")
    parser.add_argument(
        "--plot_type", type=str, help="The type of plot you wanna visualize"
    )
    return parser


def plot(outcome_dir, plot_type):
    outcome_files = glob.glob(f"{outcome_dir}/*train_discriminator_.csv")
    outcome = pd.concat(
        [pd.read_csv(outcome_file, delimiter=",") for outcome_file in outcome_files]
    )
    if plot_type == "A":
        y_test, y_scores = list(outcome["y"].dropna()), list(
            outcome["y_prob_1"].dropna()
        )

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve, classifying training vs. generated molecules")
        plt.legend(loc="lower right")
        plt.savefig("scripts/figures/ped_fig/train_disc_line.png")
        plt.show()

    else:
        y_test, y_pred = list(outcome["y"].dropna()), list(outcome["y_pred"].dropna())

        cm = confusion_matrix(y_test, y_pred)
        # Normalize the confusion matrix to show percentages
        cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        classes = ["Known", "Generated"]
        # Plot confusion matrix
        sns.heatmap(
            cm_percentage,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title("Confusion Matrix, classifying training vs. generated molecules")
        plt.savefig("scripts/figures/ped_fig/train_disc_confusion_matrix.png")
        plt.show()


def main(args):
    plot(outcome_dir=args.outcome_dir, plot_type=args.plot_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
