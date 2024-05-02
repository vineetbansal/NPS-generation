import argparse
import numpy as np
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity, ExplicitBitVect
from tqdm import tqdm

from clm.functions import (
    compute_fingerprints,
    set_seed,
    seed_type,
    clean_mol,
    generate_df,
    get_mass_range,
    write_to_csv_file,
)

# suppress rdkit errors
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")
tqdm.pandas()
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
        "--carbon_file",
        type=str,
        default=None,
        help="Path to the training dataset altered by the add carbon step.",
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


def get_fp_obj(fp_string, bits=1024):
    # Get FingerPrint object from a Base64 fingerprint string
    # There currently seems to be no more concise way to do this since
    # .FromBase64 doesn't return the modified object
    fp = ExplicitBitVect(bits)
    fp.FromBase64(fp_string)
    return fp


def get_inchikey(smile):
    # Get Inchikey for a valid smile
    return Chem.inchi.MolToInchiKey(clean_mol(smile, raise_error=True))


def match_molecules(row, dataset, data_type):
    match = dataset[dataset["mass"].between(row["mass_range"][0], row["mass_range"][1])]

    # For the PubChem dataset, not all SMILES might be valid; consider only the ones that are.
    # If a `fingerprint` column exists, then we have a valid SMILE
    if len(match) > 0 and data_type == "PubChem" and "fingerprint" not in dataset:
        match = match[
            match.apply(
                lambda x: clean_mol(x["smiles"], raise_error=False) is not None, axis=1
            )
        ]

    # Assign sequential ranks based on decreasing frequency of sampling
    match = (
        match.sort_values("size", ascending=False)
        if data_type == "model"
        else match.sample(frac=1)
    )
    match = match.assign(rank=np.arange(match.shape[0]))
    # We'll be adding columns to `match` later. For now keep track of
    # what the `target_<rank/smiles/mass/formula>` values are for the given
    # source (model/train/PubChem)
    match.columns = "target_" + match.columns

    # For Pubchem, Inchi keys may not have been pre-calculated
    if data_type == "PubChem" and "target_inchikey" not in match.columns:
        match["target_inchikey"] = match["target_smiles"].apply(
            lambda x: get_inchikey(x)
        )

    rank = match[match["target_inchikey"] == row["inchikey"]][
        ["target_size", "target_rank", "target_source"]
    ]

    # `rank` denotes the best match
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
    # `n_candidates` is the number of candidates close enough to the NPS
    # in terms of molecular mass.
    rank = rank.assign(n_candidates=match.shape[0])

    tc = match
    if tc.shape[0] > 1:
        tc = tc.head(1)
    if data_type == "model" and match.shape[0] > 1:
        # For the generative model, we'll pick a molecule sampled less
        # frequently against which to compare fingerprints.
        tc = pd.concat([tc, match.tail(-1).sample()])

    if tc.shape[0] > 0:
        if "target_fingerprint" in tc:
            target_fps = [get_fp_obj(fp) for fp in tc["target_fingerprint"]]
        else:
            target_mols = [
                clean_mol(smile, selfies=False) for smile in tc["target_smiles"].values
            ]
            target_fps = compute_fingerprints(target_mols, algorithm="ecfp6")
        tc["Tc"] = [
            FingerprintSimilarity(row["fp"], target_fp) for target_fp in target_fps
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
    carbon_file=None,
):
    set_seed(seed)

    train = generate_df(train_file, chunk_size)
    train = train.assign(size=np.nan)

    test = generate_df(test_file, chunk_size)
    test["fp"] = test.apply(
        lambda row: AllChem.GetMorganFingerprintAsBitVect(
            clean_mol(row["smiles"]), 3, nBits=1024
        ),
        axis=1,
    )
    test["mass_range"] = test.apply(
        lambda x: get_mass_range(x["mass"], err_ppm), axis=1
    )
    test = test.assign(mass_known=test["mass"].isin(train["mass"]))
    test = test.assign(formula_known=test["formula"].isin(train["formula"]))

    logger.info("Reading PubChem file")
    pubchem = pd.read_csv(pubchem_file, delimiter="\t", header=None)

    # PubChem tsv can have 3, 4 or 5 columns
    match len(pubchem.columns):
        case 3:
            pubchem.columns = ["smiles", "mass", "formula"]
        case 4:
            pubchem.columns = ["smiles", "mass", "formula", "fingerprint"]
            pubchem = pubchem.dropna(subset="fingerprint")
        case 5:
            pubchem.columns = ["smiles", "mass", "formula", "fingerprint", "inchikey"]
            pubchem = pubchem.dropna(subset="fingerprint")
        case _:
            raise RuntimeError("Unexpected column count for PubChem")

    pubchem = pubchem.assign(size=np.nan)

    logger.info("Reading sample file from generative model")
    gen = pd.read_csv(sample_file)

    inputs = {
        "model": gen.assign(source="model"),
        "PubChem": pubchem.assign(source="PubChem"),
        "train": train.assign(source="train"),
    }

    if carbon_file:
        addcarbon = pd.read_csv(carbon_file, delimiter=r"\s")

        # Rename carbon's column names to coincide with other inputs and drop input smiles
        addcarbon.rename(columns={"mutated_smiles": "smiles"}, inplace=True)
        addcarbon.drop(columns="input_smiles", inplace=True)

        # assign frequencies of add carbons as none
        addcarbon = addcarbon.assign(size=np.nan)

        inputs["addcarbon"] = addcarbon.assign(source="addcarbon")

    rank_df, tc_df = pd.DataFrame(), pd.DataFrame()
    for datatype, dataset in inputs.items():
        logging.info(f"Generating statistics for model {datatype}")

        results = test.progress_apply(
            lambda x: match_molecules(x, dataset, datatype), axis=1
        )

        logging.info(f"Generated statistics for model {datatype}")
        rank = pd.concat(results[0].to_list())
        rank.insert(0, "Index", range(len(rank)))
        tc = pd.concat(results[1].to_list())
        tc.insert(0, "Index", range(len(tc)))
        rank_df = pd.concat([rank_df, rank])
        tc_df = pd.concat([tc_df, tc])

    write_to_csv_file(ranks_file, rank_df)
    write_to_csv_file(
        tc_file, tc_df.drop("target_fingerprint", axis=1, errors="ignore")
    )


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
        carbon_file=args.carbon_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
