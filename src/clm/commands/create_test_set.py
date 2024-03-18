import argparse
import logging
import numpy as np
from selfies import encoder as selfies_encoder
from selfies.exceptions import EncoderError
from clm.functions import read_file, write_smiles, seed_type
from clm.commands.create_training_sets import get_similar_smiles


logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--input-file", type=str, required=True, help="File path of smiles file"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="Output test smiles file path ({fold} in path is populated automatically)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of CV Folds to generate for train/test split (default %(default)s",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="SMILES",
        help="Representation (one of SMILES/SELFIES)",
    )
    parser.add_argument(
        "--min-tc",
        type=float,
        default=0,
        help="Minimum fingerprint similarity (Tanimoto Coefficient) to seed molecule. 0 for no similarity requirement",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=100,
        help="Number of molecules to generate for each seed molecule",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=200,
        help="Maximum tries to get n_molecules with min_tc",
    )
    parser.add_argument(
        "--seed", type=seed_type, nargs="?", default=None, help="Random Seed"
    )
    parser.add_argument(
        "--max-input-smiles",
        type=int,
        default=None,
        help="Maximum smiles to read from input file (useful for testing)",
    )

    return parser


def create_test_set(
    input_file=None,
    test_file=None,
    folds=10,
    representation="SMILES",
    min_tc=0,
    n_molecules=100,
    max_tries=200,
    seed=None,
    max_input_smiles=None,
):
    logger.info("reading input SMILES ...")
    smiles = read_file(smiles_file=input_file, max_lines=max_input_smiles)

    if min_tc > 0:
        logger.info(f"picking {n_molecules} molecules with min_tc={min_tc} ...")
        smiles = get_similar_smiles(
            smiles,
            min_tc=min_tc,
            n_molecules=n_molecules,
            max_tries=max_tries,
            representation=representation,
        )

    np.random.seed(seed)
    np.random.shuffle(smiles)
    folds = np.array_split(smiles, folds)
    test = folds[0]

    if representation == "SELFIES":
        logger.info("converting SMILES strings to SELFIES ...")

        test_out = []
        for sm in test:
            try:
                sf = selfies_encoder(sm)
                test_out.append(sf)
            except EncoderError:
                pass
        test = test_out

    write_smiles(test, str(test_file))


def main(args):
    create_test_set(
        input_file=args.input_file,
        test_file=args.test_file,
        folds=args.folds,
        representation=args.representation,
        min_tc=args.min_tc,
        n_molecules=args.n_molecules,
        max_tries=args.max_tries,
        seed=args.seed,
        max_input_smiles=args.max_input_smiles,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
