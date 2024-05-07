import argparse
import numpy as np
from rdkit.DataStructs import FingerprintSimilarity
from clm.functions import (
    clean_mol,
    read_file,
    write_to_csv_file,
    compute_fingerprint,
    read_csv_file,
)
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


def find_max_similarity_fingerprint(target_smile, ref_smiles, ref_fps):
    target_fps = calculate_fingerprint(target_smile)

    if target_fps is None:
        return None, None

    tcs = [FingerprintSimilarity(target_fps, ref_fp) for ref_fp in ref_fps]

    return np.max(tcs), ref_smiles[np.argmax(tcs)]


def write_nn_Tc(query_file, reference_file, output_file):
    ref_fps, ref_smiles = [], []
    for smile in read_file(reference_file, stream=True, smile_only=True):
        if (fps := calculate_fingerprint(smile)) is not None:
            ref_fps.append(fps)
            ref_smiles.append(smile)

    total_lines = sum(1 for _ in open(query_file, "r"))
    n_processed = 0
    for query in read_csv_file(query_file, chunksize=10000):
        results = query["smiles"].apply(
            lambda x: find_max_similarity_fingerprint(x, ref_smiles, ref_fps)
        )
        query = query.assign(nn_tc=[i[0] for i in results])
        query = query.assign(nn=[i[1] for i in results])

        write_to_csv_file(
            output_file, info=query, mode="w" if n_processed == 0 else "a+"
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
