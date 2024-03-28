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
    return None


def train_discriminator(train_file, sample_file, max_mols, output_file, seed):
    set_seed(seed)

    train_smiles = read_file(train_file, smile_only=True)
    gen_smiles = read_file(sample_file, smile_only=True)

    gen_smiles = np.unique(gen_smiles)
    novel_smiles = [sm for sm in gen_smiles if sm not in set(train_smiles)]

    # Sample a random choice of molecules
    if len(novel_smiles) > max_mols and len(train_smiles) > max_mols:
        novel_smiles = np.random.choice(novel_smiles, max_mols)
        train_smiles = np.random.choice(train_smiles, max_mols)
    elif len(novel_smiles) > len(train_smiles):
        novel_smiles = np.random.choice(novel_smiles, len(train_smiles))

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
        max_mols=args.max_mols,
        output_file=args.output_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
