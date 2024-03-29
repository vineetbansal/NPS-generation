import argparse
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)


def add_args(parser):
    parser.add_argument("--sample_file", type=str, help="Path to the sampled file")
    parser.add_argument("--train_file", type=str, help="Path to the train file")
    parser.add_argument("--sample_no", type=int, help="Number of samples to select")
    parser.add_argument("--pubchem_file", type=str, help="Path to the PubChem file")
    parser.add_argument(
        "--output_file", type=str, help="Path to the save the output file"
    )

    return parser


def write_outcome_distr(sample_file, sample_no, train_file, pubchem_file, output_file):
    sample_file = pd.read_csv(sample_file, delimiter=",")
    sample = sample_file.sample(
        n=sample_no, replace=True, weights=sample_file["size"], ignore_index=True
    )
    sample.assign(source="model")
    pubchem = pd.read_csv(
        pubchem_file, delimiter="\t", header=None, names=["smiles", "mass", "formula"]
    )
    formulas = set(sample.formula)
    pubchem = pubchem[pubchem["formula"].isin(formulas)]
    pubchem.assign(source="pubchem")
    train = pd.read_csv(train_file)
    train.assign(source="train")
    combination = pd.concat([sample, pubchem, train])
    combination.to_csv(output_file, index=False)


def main(args):
    write_outcome_distr(
        sample_file=args.sample_file,
        sample_no=args.sample_no,
        train_file=args.train_file,
        pubchem_file=args.pubchem_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
