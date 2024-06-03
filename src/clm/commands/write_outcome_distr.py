import pandas as pd
from clm.functions import (
    read_file,
    write_to_csv_file,
    read_csv_file,
    set_seed,
    seed_type,
)


def add_args(parser):
    parser.add_argument("--sample_file", type=str, help="Path to the sampled file")
    parser.add_argument("--train_file", type=str, help="Path to the train file")
    parser.add_argument("--max_mols", type=int, help="Number of samples to select")
    parser.add_argument("--pubchem_file", type=str, help="Path to the PubChem file")
    parser.add_argument(
        "--output_file", type=str, help="Path to the save the output file"
    )
    parser.add_argument(
        "--seed",
        type=seed_type,
        default=None,
        nargs="?",
        help="Random seed for reproducibility",
    )
    return parser


def write_outcome_distr(
    sample_file, max_mols, train_file, pubchem_file, output_file, seed=None
):
    set_seed(seed)

    sample = read_csv_file(sample_file, delimiter=",")
    if sample.shape[0] > max_mols:
        sample = sample.sample(
            n=max_mols, replace=True, weights=sample["size"], ignore_index=True
        )

    pubchem = read_csv_file(
        pubchem_file, delimiter="\t", header=None, names=["smiles", "mass", "formula"]
    )
    pubchem = pubchem[pubchem["formula"].isin(set(sample.formula))]
    pubchem = pubchem.drop_duplicates(subset=["formula"], keep="first")

    train = pd.DataFrame({"smiles": read_file(train_file, smile_only=True)})
    combination = pd.concat(
        [
            sample.assign(source="model"),
            pubchem.assign(source="pubchem"),
            train.assign(source="train"),
        ]
    )

    write_to_csv_file(output_file, combination, columns=["smiles", "source"])
    return combination[["smiles", "source"]]


def main(args):
    write_outcome_distr(
        sample_file=args.sample_file,
        max_mols=args.max_mols,
        train_file=args.train_file,
        pubchem_file=args.pubchem_file,
        output_file=args.output_file,
        seed=args.seed,
    )
