import argparse
import os
import pandas as pd
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


def function(df, split_data):
    current = pd.DataFrame([df])
    current_formula = current["formula"].values[0]
    if current_formula not in split_data.keys():
        return current.assign(source="DeepMet")
    match = split_data[current_formula].sample(n=1)
    return pd.concat([current.assign(source="DeepMet"), match.assign(source="PubChem")])


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

    # Every formula in the key should be unique here
    split_data = {formula: df for formula, df in pubchem.groupby("formula")}

    result = sample.apply(lambda x: function(x, split_data), axis=1)
    matches = pd.concat(result.to_list())
    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    matches.to_csv(output_file, index=False)


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
