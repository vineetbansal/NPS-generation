import argparse
import numpy as np
import pandas as pd
from rdkit.DataStructs import FingerprintSimilarity
from clm.functions import clean_mol, compute_fingerprint
import os
import logging


logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--query_file", type=str, help="Path to the prep file ")
    parser.add_argument("--reference_file", type=str, help="Path to the PubChem file")
    parser.add_argument("--output_file", type=str, help="Path to save the output file")

    return parser


def calculate_fingerprint(smile):
    if (mol := clean_mol(smile, raise_error=False)) is not None:
        return compute_fingerprint(mol)
    return None


def find_max_similarity_fingerprint(target_row, ref_smiles, ref_fps, ref_inchikeys):
    target_fps = calculate_fingerprint(target_row.smiles)
    try:
        target_inchikey = target_row.inchikey
    except AttributeError:
        target_inchikey = target_row.target_inchikey

    if target_fps is None:
        return None, None

    tcs = []
    for ref_smile, ref_fp, ref_inchikey in zip(ref_smiles, ref_fps, ref_inchikeys):
        # Avoid comparing tcs of exactly same molecule
        if not (target_inchikey == ref_inchikey):
            tcs.append(FingerprintSimilarity(target_fps, ref_fp))
        else:
            tcs.append(-1)

    return np.max(tcs), ref_smiles[np.argmax(tcs)]


def write_nn_Tc(query_file, reference_file, output_file):
    ref_fps, ref_smiles, ref_inchikeys = [], [], []
    for row in pd.read_csv(reference_file, chunksize=1, iterator=True):
        if (fps := calculate_fingerprint(row.smiles.values[0])) is not None:
            ref_fps.append(fps)
            ref_smiles.append(row.smiles.values[0])
            try:
                ref_inchikeys.append(row.inchikey.values[0])
            except AttributeError:
                ref_inchikeys.append(row.target_inchikey.values[0])

    total_lines = sum(1 for _ in open(query_file, "r"))
    n_processed = 0
    for query in pd.read_csv(query_file, chunksize=10000):
        results = query.apply(
            lambda x: find_max_similarity_fingerprint(
                x, ref_smiles, ref_fps, ref_inchikeys
            ),
            axis=1,
        )
        query = query.assign(nn_tc=[i[0] for i in results])
        query = query.assign(nn=[i[1] for i in results])

        query.to_csv(
            output_file,
            mode="a+",
            index=False,
            header=not os.path.exists(output_file),
            compression="gzip" if str(output_file).endswith(".gz") else None,
        )

        n_processed += len(query)
        logger.info(f"Processed {n_processed}/{total_lines}")


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
