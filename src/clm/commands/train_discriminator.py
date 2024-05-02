import argparse
import numpy as np
import os
import pandas as pd
from rdkit import Chem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from clm.functions import set_seed, seed_type, clean_mol, write_to_csv_file


def add_args(parser):
    parser.add_argument(
        "--train_file", type=str, help="Training csv file with smiles as a column."
    )
    parser.add_argument(
        "--sampled_file",
        type=str,
        help="Sampled csv file with smiles as a column, or a text file with one SMILES per line.",
    )
    parser.add_argument(
        "--max_mols",
        type=int,
        default=100_000,
        help="Total number of molecules to sample.",
    )
    parser.add_argument("--output_file", type=str)
    parser.add_argument(
        "--seed", type=seed_type, default=None, nargs="?", help="Random seed"
    )
    return parser


def create_output_dir(output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass


def calculate_fingerprint(smile):
    if (mol := clean_mol(smile, raise_error=False)) is not None:
        return Chem.RDKFingerprint(mol)


def train_discriminator(train_file, sample_file, output_file, seed, max_mols=100_000):
    set_seed(seed)

    train_smiles = pd.read_csv(train_file)
    sample_smiles = pd.read_csv(sample_file)

    sample_smiles = sample_smiles[
        ~sample_smiles["inchikey"].isin(train_smiles["inchikey"])
    ]

    train_smiles = train_smiles.smiles
    train_smiles = (
        np.random.choice(train_smiles, size=max_mols, replace=False)
        if len(train_smiles) > max_mols
        else train_smiles.to_numpy()
    )

    # Match the number of novel and train smiles
    if sample_smiles.shape[0] > len(train_smiles):
        sample_smiles.sample(
            n=len(train_smiles), weights="size", random_state=seed, replace=False
        )

    sample_smiles = sample_smiles.smiles.to_numpy()

    np_fps = []
    labels = []
    for idx, smile in tqdm(
        enumerate(np.concatenate((train_smiles, sample_smiles), axis=0))
    ):
        if (fp := calculate_fingerprint(smile)) is not None:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            np_fps.append(arr)

            labels.append(1) if idx < len(train_smiles) else labels.append(0)

    # Split into train/test folds
    X_train, X_test, y_train, y_test = train_test_split(
        np_fps, labels, test_size=0.2, random_state=0
    )

    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)

    # Predict classes for held-out molecules
    y_pred = rf.predict(X_test)
    y_probs = rf.predict_proba(X_test)
    y_prob_1 = [x[1] for x in y_probs]

    output_dict = {
        "y": y_test,
        "y_pred": y_pred,
        "y_prob_1": y_prob_1,
    }
    output_df = pd.DataFrame(output_dict)

    # Create an output directory if it doesn't exist already
    create_output_dir(output_file)

    write_to_csv_file(output_file, output_df)
    return output_df


def main(args):
    train_discriminator(
        train_file=args.train_file,
        sample_file=args.sampled_file,
        output_file=args.output_file,
        seed=args.seed,
        max_mols=args.max_mols,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
