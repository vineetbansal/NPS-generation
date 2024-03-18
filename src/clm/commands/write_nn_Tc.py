import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from clm.functions import clean_mol
from tqdm import tqdm

tqdm.pandas()


def add_args(parser):
    parser.add_argument("--query_file", type=str)
    parser.add_argument("--reference_file", type=str)
    parser.add_argument("--output_file", type=str)

    return parser


def calculate_fingerprint(smile):
    if (mol := clean_mol(smile, raise_error=False)) is not None:
        return Chem.RDKFingerprint(mol)
    return None


def find_max_similarity_fingerprint(target_smile, ref_smiles, ref_fps):
    target_fps = calculate_fingerprint(target_smile)

    if target_fps is None:
        return None

    tcs = [FingerprintSimilarity(target_fps, ref_fp) for ref_fp in ref_fps if ref_fp is not None]
    return np.max(tcs), ref_smiles[np.argmax(tcs)]


def write_nn_Tc(query_file, reference_file, output_file):
    query = pd.read_csv(query_file)
    ref = pd.read_csv(reference_file)

    ref["fps"] = ref["smiles"].apply(calculate_fingerprint)
    ref_smiles = [ref["smiles"].values[i] for i, fp in enumerate(ref["fps"]) if fp is not None]

    results = query["smiles"].apply(
        lambda x: find_max_similarity_fingerprint(x, ref_smiles, ref["fps"])
    )

    query["nn_tc"] = [i[0] for i in results]
    query["nn"] = [i[1] for i in results]

    query.to_csv(output_file,
                 index=False,
                 compression="gzip" if str(output_file).endswith(".gz") else None,
                 )
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
