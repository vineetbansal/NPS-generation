import argparse
import pandas as pd
import os
from clm.functions import set_seed, seed_type

parser = argparse.ArgumentParser(description=__doc__)


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
    sample = pd.read_csv(sample_file, delimiter=",")
    if len(sample) > sample_no:
        sample = sample.sample(
            n=sample_no, replace=True, weights=sample["size"], ignore_index=True
        )
    pubchem = pd.read_csv(pubchem_file, delimiter="\t", header=None)

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

    pubchem = pubchem[pubchem["formula"].isin(set(sample.formula))]
    pubchem = pubchem.drop_duplicates(subset=["formula"], keep="first")

    combination = pd.concat(
        [
            sample.assign(source="DeepMet"),
            pubchem.assign(source="PubChem"),
        ]
    )

    # Make an output directory if it doesn't yet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combination.to_csv(output_file, index=False)
    return combination


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
