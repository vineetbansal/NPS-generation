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
    parser.add_argument("--query_file", type=str)
    parser.add_argument("--reference_file", type=str)
    parser.add_argument("--output_file", type=str)

    return parser


def write_nn_Tc(query_file, reference_file, output_file):
    # read the query set
    query = pd.read_csv(query_file)

    # read the reference set
    ref = pd.read_csv(reference_file)

    # calculate the masses and molecular formulas of the query set
    ref_smiles = ref["smiles"].values
    ref_mols = clean_mols(ref_smiles)
    # remove invalid molecules
    ref_smiles = [
        ref_smiles[idx] for idx, mol in enumerate(ref_mols) if mol is not None
    ]
    ref_mols = [ref_mols[idx] for idx, mol in enumerate(ref_mols) if mol is not None]
    # compute fingerprints
    ref_fps = get_rdkit_fingerprints(ref_mols)

    # compute nearest-neighbor Tc between training and test sets
    ncol = len(list(query))
    query[["nn_tc"]] = np.nan
    query[["nn"]] = ""
    counter = 0
    for row in tqdm(query.itertuples(), total=query.shape[0]):
        try:
            query_mol = clean_mol(row.smiles)
            query_fp = Chem.RDKFingerprint(query_mol)
            tcs = [FingerprintSimilarity(query_fp, ref_fp) for ref_fp in ref_fps]
            max = np.max(tcs)
            # also get the nearest neighbor
            nn = ref_smiles[np.argmax(tcs)]
            # assign to the test data frame
            query.iat[counter, ncol] = max
            query.iat[counter, ncol + 1] = nn
        except ValueError as e:
            print(e)
        # increment counter
        counter = counter + 1

    # write to output file
    query.to_csv(output_file, index=False, compression="gzip" if str(output_file).endswith(".gz") else None,)
    return query


def main(args):
    write_nn_Tc(
        query_file=args.query_file,
        reference_file=args.reference_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
