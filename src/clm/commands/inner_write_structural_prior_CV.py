import argparse
import numpy as np
import logging
import os
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.DataStructs import FingerprintSimilarity
from tqdm import tqdm

from clm.functions import (
    get_ecfp6_fingerprints,
    read_file,
    set_seed,
    seed_type,
    clean_mols,
    clean_mol,
)

# suppress rdkit errors
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")
logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--ranks_file", type=str)
    parser.add_argument("--tc_file", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--pubchem_file", type=str)
    parser.add_argument("--sample_file", type=str)
    parser.add_argument("--err_ppm", type=int)
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument(
        "--seed", type=seed_type, default=None, nargs="?", help="Random seed"
    )
    return parser


def write_to_file(file_name, df):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_csv(
        file_name,
        index=False,
        compression="gzip" if str(file_name).endswith(".gz") else None,
    )


def generate_df(smiles_file, chunk_size):
    smiles = read_file(smiles_file)
    df = pd.DataFrame(columns=["smiles", "mass", "formula"])

    for i in tqdm(range(0, len(smiles), chunk_size)):
        mols = clean_mols(
            smiles[i : i + chunk_size],
            selfies=False,
            disable_progress=True,
            return_dict=True,
        )

        chunk_data = [
            {
                "smiles": smile,
                "mass": round(Descriptors.ExactMolWt(mol), 4),
                "formula": rdMolDescriptors.CalcMolFormula(mol),
            }
            for smile, mol in mols.items()
            if mol
        ]

        if chunk_data:
            df = pd.concat([df, pd.DataFrame(chunk_data)], ignore_index=True)

    return df


def get_mass_range(mass, err_ppm):
    min_mass = (-err_ppm / 1e6 * mass) + mass
    max_mass = (err_ppm / 1e6 * mass) + mass

    return min_mass, max_mass


def match_molecules(row, test, dataset, data_type):
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
        target_mols = test[test["smiles"] == tc["target_smiles"].values[0]]["mol"]
        target_fps = get_ecfp6_fingerprints(target_mols)
        query_fp = AllChem.GetMorganFingerprintAsBitVect(row["mol"], 3, nBits=1024)
        tcs = [FingerprintSimilarity(query_fp, target_fp) for target_fp in target_fps]
        tc["Tc"] = pd.Series(tcs)
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

    tc = pd.concat(
        [
            pd.DataFrame([row[:-2]])
            .iloc[np.full(tc.shape[0], 0)]
            .reset_index(drop=True),
            tc.reset_index(drop=True),
        ],
        axis=1,
    )
    rank = pd.concat(
        [pd.DataFrame([row[:-2]]).reset_index(drop=True), rank.reset_index(drop=True)],
        axis=1,
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

        results = test.apply(
            lambda x: match_molecules(x, test, dataset, datatype), axis=1
        )

        rank_df = pd.concat([rank_df, pd.concat(results[0].to_list())])
        tc_df = pd.concat([tc_df, pd.concat(results[1].to_list())])

    write_to_file(ranks_file, rank_df)
    write_to_file(tc_file, tc_df)


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
