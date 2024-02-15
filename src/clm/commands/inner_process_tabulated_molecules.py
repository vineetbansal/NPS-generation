# Rscript R/sample-molecules/inner-write-uniq-CV.R
#   --input_file /media/vineetb/T7/projects/nps/NPS-generation/snakemake/data/prior/samples/LOTUS_SMILES_\$_unique_masses.csv
#   --cv_file /media/vineetb/T7/projects/nps/NPS-generation/snakemake/data/prior/inputs/train_LOTUS_SMILES_seed0_\$.smi
#   --output_file /media/vineetb/T7/projects/nps/NPS-generation/snakemake/data/out.csv
#   --summary_fn freq_avg

# smiles,mass,formula,size
# CCC(CCC(C)C1CCC2C3CC=C4CC(OC5OC(CO)C(O)C(O)C5O)CCC4(C)C3CCC12C)C(C)C,576.43899,C35H60O6,0.47750578812247
# Cc1cc(O)c2c(c1)C(=O)c1cc(O)cc(O)c1C2=O,270.052823,C15H10O5,0.322933971820781
# CN1CCc2cc3c(c4c2C1Cc1ccccc1-4)OCO3,279.125929,C18H17NO2,0.281061023213918
# COc1c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c2c1=O,316.058303,C16H12O7,0.267757011308114
# CSCCC(N)C(=O)O,149.05105,C5H11NO2S,0.265365175923131


import argparse
import os
import pandas as pd
import numpy as np


# Argument parser setup
def add_args(parser):
    parser.add_argument(
        "--input_file", type=str, nargs="+", help="Path to the input CSV file."
    )
    parser.add_argument("--cv_file", type=str, nargs="+", help="Path to the CV file.")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file.")
    parser.add_argument(
        "--summary_fn",
        type=str,
        default="freq_avg",
        help="Summary function (fp10k/freq_sum/freq_avg).",
    )
    return parser


def process_tabulated_molecules(input_file, cv_file, output_file, summary_fn):
    _metas = []
    for fold_idx, file in enumerate(input_file):
        _meta = pd.read_csv(file, dtype={"smiles": str})
        _meta["fold"] = fold_idx
        _metas.append(_meta)
    meta = pd.concat(_metas)

    data = meta.pivot_table(
        index="smiles", columns="fold", values="size", aggfunc="first", fill_value=0
    )
    data[np.isnan(data)] = 0

    uniq_smiles = data.index.to_numpy()

    # Read training set SMILES
    for fold_idx, cv_file in enumerate(cv_file):
        cv_dat = pd.read_csv(cv_file, names=["smiles"]).query("smiles in @uniq_smiles")
        # censor these values
        if len(cv_dat) > 0:
            data[cv_dat["smiles"], fold_idx] = np.nan

    # Optionally normalize by total sampling frequency
    if summary_fn == "fp10k":
        data = 10e3 * data / np.nansum(data, axis=0)

    # Calculate mean/sum
    if summary_fn == "freq-sum":
        # With what frequency (across all folds)
        # were valid molecules produced by our models?
        sums = np.nansum(data, axis=1)
        data = pd.DataFrame({"smiles": list(uniq_smiles), "size": sums})
    else:
        # With what average frequency (across all folds)
        # were valid molecules produced by our models?
        means = np.nanmean(data, axis=1)
        data = pd.DataFrame({"smiles": list(uniq_smiles), "size": means})

    # Arrange by size
    data = data.sort_values(by="size", ascending=False).query("size > 0")

    # Add metadata (mass and formula)
    data = data.merge(meta[["smiles", "mass", "formula"]], how="left", on="smiles")

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(output_file, index=False)


def main(args):
    process_tabulated_molecules(
        input_file=args.input_file,
        cv_file=args.cv_file,
        output_file=args.output_file,
        summary_fn=args.summary_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
