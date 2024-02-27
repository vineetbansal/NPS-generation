"""
Evaluate the performance of our generative model by writing molecules that
match the masses of held-out metabolites from HMDB5, along with their
ClassyFire classifications and their Tanimoto coefficients to the ground truth.
"""

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from tqdm import tqdm

from clm.functions import clean_mol, clean_mols, get_rdkit_fingerprints


def add_args(parser):
    parser.add_argument("k")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--output_file", type=str)

    return parser


def write_nn_Tc_CV(k, train_file, test_file, output_file):
    df = pd.DataFrame()
    cv_folds = range(k)
    for cv_fold in cv_folds:
        # read training and test sets
        cv_idx = str(cv_fold + 1)
        train = pd.read_csv(train_file.replace('$', cv_idx))
        test = pd.read_csv(test_file.replace('$', cv_idx))

        # create fingerprints
        train_smiles = train['smiles'].values
        train_mols = clean_mols(train_smiles)
        train_fps = get_rdkit_fingerprints(train_mols)

        # compute nearest-neighbor Tc between training and test sets
        ncol = len(list(test))
        test[['nn_tc']] = np.nan
        test[['nn']] = ""
        counter = 0
        for row in tqdm(test.itertuples(), total=test.shape[0]):
            test_mol = clean_mol(row.smiles)
            test_fp = Chem.RDKFingerprint(test_mol)
            tcs = [FingerprintSimilarity(test_fp, train_fp) for train_fp in \
                   train_fps]
            max = np.max(tcs)
            # also get the nearest neighbor
            nn = train_smiles[np.argmax(tcs)]
            # assign to the test data frame
            test.iat[counter, ncol] = max
            test.iat[counter, ncol + 1] = nn
            # increment counter
            counter = counter + 1

        test[['cv_fold']] = cv_idx
        df = df.append(test)

    # write to output file
    df.to_csv(output_file, index=False, compression='gzip')


def main(args):
    write_nn_Tc_CV(
        k=args.k,
        train_file=args.train_file,
        test_file=args.test_file,
        output_file=args.output_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
