import argparse
import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from clm.functions import set_seed, seed_type

parser = argparse.ArgumentParser(description=__doc__)
logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--sample_file", type=str, required=True, help="Path to the sampled file"
    )
    parser.add_argument(
        "--sample_no", type=int, default=500_000, help="Number of samples to select"
    )
    parser.add_argument(
        "--pubchem_file", type=str, required=True, help="Path to the PubChem file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the save the output file",
    )
    parser.add_argument(
        "--seed", type=seed_type, default=None, nargs="?", help="Random seed."
    )

    return parser


def prep_nn_tc(sample_file, sample_no, pubchem_file, output_file, seed=None):
    set_seed(seed)
    sample_file = pd.concat(
        [
            chunk
            for chunk in tqdm(
                pd.read_csv(sample_file, delimiter=",", chunksize=1000),
                desc="Loading sample data",
            )
        ]
    )

    sample = sample_file.sample(
        n=sample_no, replace=True, weights=sample_file["size"], ignore_index=True
    )
    # Save the current index values in an `id` column, since we will need
    # these for filtering later.
    sample["id"] = sample.index

    pubchem = pd.concat(
        [
            chunk
            for chunk in tqdm(
                pd.read_csv(pubchem_file, delimiter="\t", header=None, chunksize=1000),
                desc="Loading PubChem data",
            )
        ]
    )

    # PubChem tsv can have 3 or 4 columns (if fingerprints are precalculated)
    match len(pubchem.columns):
        case 3:
            pubchem.columns = ["smiles", "mass", "formula"]
        case 4:
            pubchem.columns = ["smiles", "mass", "formula", "fingerprint"]
            pubchem = pubchem.dropna(subset="fingerprint")
            # ignore the fingerprint column since we don't need it
            pubchem = pubchem.drop(columns="fingerprint")
        case _:
            raise RuntimeError("Unexpected column count for PubChem")

    logger.info("Joining sample table with PubChem on formula")
    data = pd.merge(sample, pubchem, on="formula", how="left", indicator=True)

    # In case there are duplicate matches for the (left) sample table
    # (i.e. grouping by `id` and `formula` gives multiple rows), we want to
    # select a random row from the (right) PubChem table.
    logger.info("Removing duplicates from PubChem matches")
    data = (
        data.groupby(["id", "formula"])
        .apply(lambda x: x.sample(1))
        .reset_index(drop=True)
    )

    # `indicator=True` in `pd.merge` adds a `_merge` column that we can inspect
    # to filter on rows that were present in both tables, and get the
    # smiles/mass/formula values from the right table.
    data = data[data["_merge"] == "both"][["smiles_y", "mass_y", "formula"]].rename(
        columns={"smiles_y": "smiles", "mass_y": "mass"}
    )

    # Finally, lay out the rows from the sample and PubChem tables vertically,
    # with a `source` column telling us where the data came from.
    data = pd.concat(
        [
            # values from the (left) sample table
            sample[["smiles", "mass", "formula", "size"]].assign(source="DeepMet"),
            # values from the (right) PubChem table (`size` is NaN)
            data.assign(size=np.nan, source="PubChem"),
        ],
        axis=0,
    )

    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    data.to_csv(output_file, index=False)

    return data


def main(args):
    prep_nn_tc(
        sample_file=args.sample_file,
        sample_no=args.sample_no,
        pubchem_file=args.pubchem_file,
        output_file=args.output_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
