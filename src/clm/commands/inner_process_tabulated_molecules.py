import argparse
import os
import pandas as pd
import numpy as np


def add_args(parser):
    parser.add_argument(
        "--input_file", type=str, nargs="+", help="Path to the input CSV file."
    )
    parser.add_argument(
        "--cv_file", type=str, nargs="+", help="Path to the  training sets."
    )
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file.")
    parser.add_argument(
        "--summary_fn",
        type=str,
        default="freq_avg",
        help="Summary function (fp10k/freq_sum/freq_avg).",
    )
    return parser


def process_tabulated_molecules(input_file, cv_files, output_file, summary_fn):
    meta = pd.concat(
        [
            pd.read_csv(file, dtype={"smiles": str}).assign(fold=idx)
            for idx, file in enumerate(input_file)
        ]
    )

    data = meta.pivot_table(
        index="smiles", columns="fold", values="size", aggfunc="first", fill_value=0
    )

    uniq_smiles = data.index.to_numpy()

    # TODO: is this part even necessary, isn't the filtering done in inner_tabulate_molecules?
    for fold_idx, cv_file in enumerate(cv_files):
        cv_dat = pd.read_csv(cv_file, names=["smiles"])
        cv_dat = cv_dat[cv_dat["smiles"].isin(uniq_smiles)]

        if not cv_dat.empty:
            data[cv_dat["smiles"], fold_idx] = np.nan

    # Optionally normalize by total sampling frequency
    if summary_fn == "fp10k":
        data = 10e3 * data / np.nansum(data, axis=0)

    # Calculate mean/sum
    if summary_fn == "freq-sum":
        # With what frequency (across all folds)
        # were valid molecules produced by our models?
        data = pd.DataFrame(
            {"smiles": list(uniq_smiles), "size": np.nansum(data, axis=1)}
        )
    else:
        # With what average frequency (across all folds)
        # were valid molecules produced by our models?
        data = pd.DataFrame(
            {"smiles": list(uniq_smiles), "size": np.nanmean(data, axis=1)}
        )

    data = data.sort_values(by="size", ascending=False).query("size > 0")

    if not data.empty:
        # Add metadata (mass and formula)
        data = data.merge(meta[["smiles", "mass", "formula"]], how="left", on="smiles")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data.to_csv(output_file, index=False)


def main(args):
    process_tabulated_molecules(
        input_file=args.input_file,
        cv_files=args.cv_file,
        output_file=args.output_file,
        summary_fn=args.summary_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
