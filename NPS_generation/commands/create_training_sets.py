import argparse
import logging
import itertools
from tqdm import tqdm
import numpy as np
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
from selfies import encoder as selfies_encoder
from selfies.exceptions import EncoderError
from NPS_generation.functions import read_smiles, write_smiles, clean_mols, seed_type
from NPS_generation.datasets import vocabulary_from_representation
from NPS_generation.util.SmilesEnumerator import SmilesEnumerator


logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--input-file", type=str, required=True, help="File path of smiles file"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Output training smiles file path ({fold} in path is populated automatically)",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        required=True,
        help="Output training smiles vocabulary file path ({fold} in path is populated automatically)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Output test smiles file path ({fold} in path is populated automatically)",
    )
    parser.add_argument(
        "--enum-factor",
        type=int,
        default=0,
        help="For each input smile, number of randomized smiles to generate (0 for no augmentation)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of CV Folds to generate for train/test split (default %(default)s, 0 to not generate test data)",
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


def get_similar_smiles(input_smiles, min_tc, n_molecules=100, max_tries=200):
    mols = clean_mols(input_smiles)
    input_smiles = [
        input_smiles[idx] for idx, mol in enumerate(mols) if mol is not None
    ]
    input_mols = [mol for mol in mols if mol is not None]
    logger.info(f"Calculating fingerprints for {len(input_mols)} valid molecules ...")
    input_fps = [
        AllChem.GetMorganFingerprintAsBitVect(input_mol, 3, nBits=1024)
        for input_mol in tqdm(input_mols)
    ]

    # shuffle SMILES and fingerprints
    inputs = list(zip(input_smiles, input_fps))
    np.random.shuffle(inputs)
    input_smiles, input_fps = zip(*inputs)

    # try to pick n molecules with minimum Tc to random seed molecule
    success = False
    for try_idx in range(max_tries):
        np.random.seed(try_idx)
        logger.info(
            f"picking {n_molecules} molecules with min_tc={min_tc} try #{try_idx} of {max_tries} ..."
        )
        inputs = list(zip(input_smiles, input_fps))
        np.random.shuffle(inputs)
        input_smiles, input_fps = zip(*inputs)

        # pick our seed molecule at random
        target_fp = input_fps[0]

        tcs = [FingerprintSimilarity(input_fp, target_fp) for input_fp in input_fps]
        # subset SMILES based on fingerprint similarity
        subset_smiles = [
            input_smiles for input_smiles, tc in zip(input_smiles, tcs) if tc >= min_tc
        ]

        # break if we have enough molecules
        if len(subset_smiles) >= n_molecules:
            subset_smiles = subset_smiles[: int(n_molecules)]
            success = True
            break

    if not success:
        raise RuntimeError(
            f"Unable to pick {n_molecules} molecules with min_tc={min_tc}"
        )

    return subset_smiles


def create_training_sets(
    input_file=None,
    train_file=None,
    test_file=None,
    vocab_file=None,
    folds=10,
    enum_factor=0,
    representation="SMILES",
    min_tc=0,
    n_molecules=100,
    max_tries=200,
    seed=None,
    max_input_smiles=None,
):
    logger.info("reading input SMILES ...")
    smiles = read_smiles(smiles_file=input_file, max_lines=max_input_smiles)

    if min_tc > 0:
        logger.info(f"picking {n_molecules} molecules with min_tc={min_tc} ...")
        smiles = get_similar_smiles(
            smiles, min_tc=min_tc, n_molecules=n_molecules, max_tries=max_tries
        )

    generate_test_data = folds > 0
    if generate_test_data:
        np.random.seed(seed)
        np.random.shuffle(smiles)
        folds = np.array_split(smiles, folds)
    else:
        folds = [smiles]

    if enum_factor > 0:
        sme = SmilesEnumerator(canonical=False, enum=True)
        for idx, fold in enumerate(folds):
            enum = []
            max_tries = 200  # randomized SMILES to generate for each input structure
            for sm_idx, sm in enumerate(tqdm(fold)):
                tries = []
                for try_idx in range(max_tries):
                    try:
                        this_try = sme.randomize_smiles(sm)
                        tries.append(this_try)
                        tries = [rnd for rnd in np.unique(tries)]
                        if len(tries) > enum_factor:
                            tries = tries[:enum_factor]
                            break
                    except AttributeError:
                        continue
                enum.extend(tries)
            folds[idx] = enum

    for fold in range(len(folds)):
        if generate_test_data:
            test = folds[fold]
            train = folds[:fold] + folds[fold + 1 :]
            train = list(itertools.chain.from_iterable(train))
        else:
            train = folds[0]
            test = None

        if representation == "SELFIES":
            logger.info("converting SMILES strings to SELFIES ...")
            train_out = []
            for sm in train:
                try:
                    sf = selfies_encoder(sm)
                    train_out.append(sf)
                except EncoderError:
                    pass
            train = train_out

            if test is not None:
                test_out = []
                for sm in test:
                    try:
                        sf = selfies_encoder(sm)
                        test_out.append(sf)
                    except EncoderError:
                        pass
                test = test_out

        write_smiles(train, train_file.format(fold=fold))
        vocabulary = vocabulary_from_representation(representation, train)
        logger.info("vocabulary of {} characters".format(len(vocabulary)))
        vocabulary.write(output_file=vocab_file.format(fold=fold))
        if test is not None:
            write_smiles(test, test_file.format(fold=fold))


def main(args):
    create_training_sets(
        input_file=args.input_file,
        train_file=args.train_file,
        test_file=args.test_file,
        vocab_file=args.vocab_file,
        folds=args.folds,
        enum_factor=args.enum_factor,
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
