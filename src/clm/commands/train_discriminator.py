import argparse
import numpy as np
import os
import pandas as pd
from rdkit import Chem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from clm.functions import read_file, set_seed, seed_type, clean_mol


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
        default=50000,
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

    train_smiles = set(read_file(train_file, smile_only=True))
    sample_smiles_gen = read_file(sample_file, smile_only=True, stream=True)

    novel_smiles = set()
    for sample_smile in sample_smiles_gen:
        if len(novel_smiles) >= max_mols:
            break
        if sample_smile not in train_smiles:
            novel_smiles.add(sample_smile)

    novel_smiles = np.array(list(novel_smiles))
    train_smiles = (
        np.random.choice(list(train_smiles), max_mols)
        if len(train_smiles) > max_mols
        else np.array(list(train_smiles))
    )

    np_fps = []
    for smile in tqdm(np.concatenate((train_smiles, novel_smiles), axis=0)):
        if (fp := calculate_fingerprint(smile)) is not None:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            np_fps.append(arr)

    labels = [1] * len(train_smiles) + [0] * len(novel_smiles)

    # Split into train/test folds
    X_train, X_test, y_train, y_test = train_test_split(
        np_fps, labels, test_size=0.2, random_state=0
    )

    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)

    # Predict classes for held-out molecules
    y_pred = rf.predict(X_test)
    y_probs = rf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs[:, 1])
    auprc = average_precision_score(y_test, y_probs[:, 1])

    y_prob_1 = [x[1] for x in y_probs]
    output_dict = {
        "y": y_test,
        "y_pred": y_pred,
        "y_prob_1": y_prob_1,
        "score": ["accuracy", "auroc", "auprc"],
        "value": [acc, auc, auprc],
    }
    output_df = pd.DataFrame(
        dict([(key, pd.Series(value)) for key, value in output_dict.items()])
    )

    # Create an output directory if it doesn't exist already
    create_output_dir(output_file)
    output_df.to_csv(output_file, index=False)
    output_df = output_df.reset_index(drop=True)
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
