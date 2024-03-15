import argparse
import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from clm.functions import clean_mol


def add_args(parser):
    parser.add_argument("--query_file", type=str)
    parser.add_argument("--reference_file", type=str)
    parser.add_argument("--output_file", type=str)

    return parser


def calculate_fingerprint(smile):
    if (mol := clean_mol(smile, raise_error=False)) is not None:
        return Chem.RDKFingerprint(mol)
    return None


def find_max_similarity_fingerprint(target_smile, query_smiles):
    highest_similarity = -1
    best_match = None  # Using None to signify no match found initially
    target_fps = calculate_fingerprint(target_smile)

    if target_fps is None:
        return None

    for i, smile in enumerate(query_smiles):
        query_fps = calculate_fingerprint(smile)

        if query_fps is None:
            continue  # Skip this iteration if the fingerprint can't be calculated

        similarity_score = FingerprintSimilarity(target_fps, query_fps)
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            best_match = (i, smile)  # Store the index and smile of the best match

    return best_match


def write_nn_Tc(query_file, reference_file, output_file):
    query = pd.read_csv(query_file)
    ref = pd.read_csv(reference_file)

    results = ref["smiles"].apply(
        lambda x: find_max_similarity_fingerprint(x, query["smiles"])
    )

    query["nn_tc"] = pd.concat(results[0].to_list())
    query["nn"] = pd.concat(results[1].to_list())

    query = query.dropna()
    query.to_csv(output_file, index=False, compression="gzip")


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
