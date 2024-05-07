import argparse
import pandas as pd
from rdkit.DataStructs import FingerprintSimilarity
from clm.functions import (
    clean_mol,
    write_to_csv_file,
    compute_fingerprint,
    read_csv_file,
)
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


def find_max_similarity_fingerprint(
    target_smiles, target_inchikey, ref_smiles, ref_fps, ref_inchikeys
):
    target_fps = calculate_fingerprint(target_smiles)

    if target_fps is None:
        return None, None

    max_tc, max_tc_ref_smile = -1, ""
    for ref_smile, ref_fp, ref_inchikey in zip(ref_smiles, ref_fps, ref_inchikeys):
        # Avoid comparing tcs of exactly same molecule
        if not (target_inchikey == ref_inchikey):
            if max_tc < (fps := FingerprintSimilarity(target_fps, ref_fp)):
                max_tc = fps
                max_tc_ref_smile = ref_smile

    return max_tc, max_tc_ref_smile


def write_nn_Tc(query_file, reference_file, output_file, query_type="model"):
    ref_fps, ref_smiles, ref_inchikeys = [], [], []

    # Processing the reference_file in chunks of one row at a time to optimize memory efficiency.
    for df in pd.read_csv(reference_file, chunksize=1, iterator=True):
        row = df.iloc[0]
        # If a particular smiles is invalid, calculate_fingerprint will return None
        if (fps := calculate_fingerprint(row.smiles)) is not None:
            ref_fps.append(fps)
            ref_smiles.append(row.smiles)
            # TODO: Reference file will always be a training set, hence will always use inchikey keyword
            # TODO: need to change test file to use training file as a reference
            try:
                ref_inchikeys.append(row.inchikey)
            except AttributeError:
                ref_inchikeys.append(row.target_inchikey)

    total_lines = sum(1 for _ in open(query_file, "r"))
    n_processed = 0
    for query in read_csv_file(query_file, chunksize=10000):
        results = query.apply(
            lambda x: find_max_similarity_fingerprint(
                x.smiles,
                x.inchikey,
                ref_smiles,
                ref_fps,
                ref_inchikeys,
            ),
            axis=1,
        )
        query = query.assign(nn_tc=[i[0] for i in results])
        query = query.assign(nn=[i[1] for i in results])

        write_to_csv_file(
            output_file, info=query, mode="a+", header=not os.path.exists(output_file)
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
