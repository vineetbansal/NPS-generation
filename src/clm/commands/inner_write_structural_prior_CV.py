import argparse
import numpy as np
import logging
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity

from clm.functions import (
    get_ecfp6_fingerprints,
    set_seed,
    seed_type,
    clean_mols,
    clean_mol,
    generate_df,
    get_mass_range,
    write_to_csv_file,
)

# suppress rdkit errors
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")
logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--ranks_file", type=str, help="Path to the rank file.")
    parser.add_argument("--tc_file", type=str, help="Path to the tc file.")
    parser.add_argument("--train_file", type=str, help="Path to the training dataset.")
    parser.add_argument("--test_file", type=str, help="Path to the test dataset.")
    parser.add_argument(
        "--pubchem_file",
        type=str,
        help="Path to the file containing PubChem information.",
    )
    parser.add_argument(
        "--sample_file", type=str, help="Path to the file containing sample molecules."
    )
    parser.add_argument(
        "--err_ppm",
        type=int,
        help="Error margin in parts per million for chemical analysis.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Size of chunks for processing large files.",
    )
    parser.add_argument(
        "--seed", type=seed_type, default=None, nargs="?", help="Random seed."
    )
    return parser


def pd_concat(row, data, col):
    return pd.concat(
        [
            pd.DataFrame([row[col]]).reset_index(drop=True),
            data.reset_index(drop=True),
        ],
        axis=1,
    )


def match_molecules(row, dataset, data_type):
    match = dataset[dataset["mass"].between(row["mass_range"][0], row["mass_range"][1])]

    match = (
        match.sort_values("size", ascending=False)
        if data_type == "model"
        else match.sample(frac=1)
    )
    match = match.assign(rank=np.arange(match.shape[0]))
    match.columns = "target_" + match.columns

    rank = match[match["target_smiles"] == row["smiles"]][
        ["target_size", "target_rank", "target_source"]
    ]

    if rank.shape[0] > 1:
        rank = rank.head(1)
    elif rank.shape[0] == 0:
        rank = pd.DataFrame(
            {
                "target_size": [np.nan],
                "target_rank": [np.nan],
                "target_source": [data_type],
            }
        )
    rank = rank.assign(n_candidates=match.shape[0])

    tc = match
    if tc.shape[0] > 1:
        tc = tc.head(1)
    if data_type == "model" and match.shape[0] > 1:
        tc = pd.concat([tc, match.tail(-1).sample()])

    if tc.shape[0] > 0:
        target_mols = clean_mols(
            tc["target_smiles"].values,
            selfies=False,
            disable_progress=True,
        )
        target_fps = get_ecfp6_fingerprints(target_mols)
        query_fp = AllChem.GetMorganFingerprintAsBitVect(row["mol"], 3, nBits=1024)
        tc["Tc"] = [
            FingerprintSimilarity(query_fp, target_fp) for target_fp in target_fps
        ]
    else:
        tc = pd.DataFrame(
            {
                "target_size": np.nan,
                "target_rank": np.nan,
                "target_source": data_type,
                "Tc": np.nan,
            },
            index=[0],
        )

    tc = pd_concat(
        row, tc, col=["smiles", "mass", "formula", "mass_known", "formula_known"]
    )
    rank = pd_concat(
        row, rank, col=["smiles", "mass", "formula", "mass_known", "formula_known"]
    )

    return pd.Series((rank, tc))


def write_structural_prior_CV(
    ranks_file,
    tc_file,
    train_file,
    test_file,
    pubchem_file,
    sample_file,
    err_ppm,
    chunk_size,
    seed,
):
    set_seed(seed)

    train = generate_df(train_file, chunk_size)
    train = train.assign(size=np.nan)

    test = generate_df(test_file, chunk_size)
    test["mol"] = test["smiles"].apply(clean_mol)
    test["mass_range"] = test.apply(
        lambda x: get_mass_range(x["mass"], err_ppm), axis=1
    )
    test = test.assign(mass_known=test["mass"].isin(train["mass"]))
    test = test.assign(formula_known=test["formula"].isin(train["formula"]))

    logger.info("Reading PubChem file")
    pubchem = pd.read_csv(
        pubchem_file, delimiter="\t", header=None, names=["smiles", "mass", "formula"]
    )
    pubchem = pubchem.assign(size=np.nan)

    logger.info("Reading sample file from generative model")
    gen = pd.read_csv(sample_file)

    inputs = {
        "model": gen.assign(source="model"),
        "PubChem": pubchem.assign(source="PubChem"),
        "train": train.assign(source="train"),
    }

    rank_df, tc_df = pd.DataFrame(), pd.DataFrame()
    for datatype, dataset in inputs.items():
        logging.info(f"Generating statistics for model {datatype}")

        results = test.apply(lambda x: match_molecules(x, dataset, datatype), axis=1)

        rank = pd.concat(results[0].to_list())
        rank.insert(0, "Index", range(len(rank)))
        tc = pd.concat(results[1].to_list())
        tc.insert(0, "Index", range(len(tc)))
        rank_df = pd.concat([rank_df, rank])
        tc_df = pd.concat([tc_df, tc])

    write_to_csv_file(ranks_file, rank_df)
    write_to_csv_file(tc_file, tc_df)


def main(args):
    write_structural_prior_CV(
        ranks_file=args.ranks_file,
        tc_file=args.tc_file,
        train_file=args.train_file,
        test_file=args.test_file,
        pubchem_file=args.pubchem_file,
        sample_file=args.sample_file,
        err_ppm=args.err_ppm,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
