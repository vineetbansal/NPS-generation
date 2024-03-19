"""
Train a model to distinguish generated versus real metabolites and report its
cross-validation accuracy.
"""

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

# import functions
from clm.functions import read_file, clean_mols, set_seed, seed_type


def add_args(parser):
    parser.add_argument(
        "--train_file", type=str, help="Training csv file with smiles as a column."
    )
    parser.add_argument(
        "--sampled_file",
        type=str,
        help="Sampled csv file with smiles as a column, or a text file with one SMILES per line.",
    )
    parser.add_argument("--output_file", type=str)
    parser.add_argument(
        "--seed", type=seed_type, default=None, nargs="?", help="Random seed"
    )
    return parser


def train_discriminator(train_file, sample_file, output_file, seed):
    set_seed(seed)
    # create output directory if it does not exist
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass

    # read SMILES from training set
    train_smiles = read_file(
        train_file, smile_only=True
    )  # (this was originally train = pd.read_csv(train_file))

    # TODO: there doesn't seem to be any training csv files
    # train_smiles = train['smiles'].values

    # read generated SMILES
    gen_smiles = read_file(sample_file, smile_only=True)
    # get unique smiles
    gen_smiles = np.unique(gen_smiles)
    # remove known ones
    gen_smiles = [sm for sm in gen_smiles if sm not in train_smiles]

    # convert to molecules
    train_mols = clean_mols(train_smiles)
    gen_mols = clean_mols(gen_smiles)
    gen_mols = [mol for mol in gen_mols if mol is not None]

    # sample a random number
    if len(gen_mols) > len(train_mols):
        gen_mols = np.random.choice(gen_mols, len(train_mols))

    # calculate fingerprints for the train and generated sets
    train_fps = [Chem.RDKFingerprint(mol) for mol in tqdm(train_mols)]
    gen_fps = [Chem.RDKFingerprint(mol) for mol in tqdm(gen_mols)]

    # merge fingerprints and create labels
    fps = train_fps + gen_fps
    labels = [1] * len(train_mols) + [0] * len(gen_mols)

    # convert to list of numpy arrays
    np_fps = []
    for fp in tqdm(fps):
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)

    # split into train/test folds
    X = np_fps
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # fit RF
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    # predict classes for held-out molecules
    y_pred = rf.predict(X_test)
    y_probs = rf.predict_proba(X_test)

    # y_pred = cross_val_predict(rf, X, y, cv=5)
    # y_prob = cross_val_predict(rf, X, y, cv=5, method='predict_proba')
    # scores = cross_validate(rf, X, y, cv=3,
    #                         scoring=('accuracy', 'roc_auc', 'average_precision'),
    #                         return_train_score=False)

    # calculate metrics
    acc = accuracy_score(y_test, y_pred)
    # b_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs[:, 1])
    auprc = average_precision_score(y_test, y_probs[:, 1])

    # create output df
    y_prob_1 = [x[1] for x in y_probs]
    output_df = pd.DataFrame({"y": y_test, "y_pred": y_pred, "y_prob_1": y_prob_1})
    score_df = pd.DataFrame(
        {"score": ["accuracy", "auroc", "auprc"], "value": [acc, auc, auprc]}
    )
    output_df = output_df._append(score_df)

    # save output
    output_df.to_csv(output_file, index=False)

    return output_df


def main(args):
    train_discriminator(
        train_file=args.train_file,
        sample_file=args.sampled_file,
        output_file=args.output_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
