import numpy as np
import os
import pandas as pd
from scipy.stats import chisquare
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)
from rdkit import rdBase
from clm.functions import set_seed, seed_type, read_csv_file

# suppress rdkit errors
rdBase.DisableLog("rdApp.error")


def add_args(parser):
    parser.add_argument(
        "--test-file", type=str, required=True, help="File path of test file"
    )
    parser.add_argument(
        "--sample-file", type=str, required=True, help="File path of sample file"
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="File path of output file"
    )
    parser.add_argument(
        "--seed",
        type=seed_type,
        default=None,
        nargs="?",
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-mols",
        type=int,
        help="Maximum number of molecules to read from sample file",
    )
    return parser


def forecast(test_file, sample_file, output_file, seed=None, max_molecules=None):

    set_seed(seed)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    test = read_csv_file(test_file)
    deepmet = read_csv_file(sample_file, nrows=max_molecules)
    deepmet = deepmet.assign(known=deepmet["inchikey"].isin(test["inchikey"]))

    # TODO: verify if this is what you're supposed to do
    deepmet = deepmet.sort_values(by=["size"], ascending=False)

    # extract ROC/PR curves
    y = deepmet["known"].tolist()
    x = deepmet["size"].tolist()
    x_rnd = deepmet.sample(frac=1)["size"].tolist()
    fpr, tpr, thresholds = roc_curve(y, x)
    roc_df = pd.DataFrame(
        {
            "mode": "true",
            "curve": "ROC",
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )
    precision, recall, thresholds = precision_recall_curve(y, x)
    pr_df = pd.DataFrame(
        {
            "mode": "true",
            "curve": "PR",
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        }
    )

    # calculate shuffled ROC/PR curves
    fpr, tpr, thresholds = roc_curve(y, x_rnd)
    roc_df_rnd = pd.DataFrame(
        {
            "mode": "random",
            "curve": "ROC",
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )
    precision, recall, thresholds = precision_recall_curve(y, x_rnd)
    pr_df_rnd = pd.DataFrame(
        {
            "mode": "random",
            "curve": "PR",
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        }
    )

    # calculate metrics
    auc = roc_auc_score(y, x)
    auprc = average_precision_score(y, x)
    # calculate shuffled metrics too
    auc_rnd = roc_auc_score(y, x_rnd)
    auprc_rnd = average_precision_score(y, x_rnd)

    # last, calculate enrichment factors
    ranks = (
        list(range(10, 100, 10))
        + list(range(100, 1000, 100))
        + list(range(1000, 10000, 1000))
        + list(range(10000, 100000, 10000))
        + list(range(100000, 1000000, 100000))
        + list(range(1000000, deepmet.shape[0], 1000000))
        + [deepmet.shape[0]]
    )
    ef = pd.DataFrame({"rank": ranks, "EF": np.nan, "n_known": 0, "pval": np.nan})
    for idx, rank in enumerate(ranks):
        obs = sum(y[0:rank])
        exp = rank * sum(y) / deepmet.shape[0]
        EF = obs / exp

        # chi-square test
        f_obs = np.array([obs, rank - obs]) / rank
        f_exp = np.array([exp, rank - exp]) / rank
        pval = chisquare(f_obs=f_obs, f_exp=f_exp).pvalue

        ef.at[idx, "EF"] = EF
        ef.at[idx, "n_known"] = obs
        ef.at[idx, "pval"] = pval

    output_df = pd.concat([roc_df, pr_df, roc_df_rnd, pr_df_rnd])
    score_df = pd.DataFrame(
        {
            "score": ["auroc", "auprc"] * 2,
            "mode": ["true"] * 2 + ["random"] * 2,
            "value": [auc, auprc, auc_rnd, auprc_rnd],
        }
    )
    output_df = output_df._append(score_df)
    output_df = output_df._append(ef)

    output_df.to_csv(output_file, index=False)


def main(args):
    forecast(
        test_file=args.test_file,
        sample_file=args.sample_file,
        output_file=args.output_file,
        seed=args.seed,
        max_molecules=args.max_mols,
    )
