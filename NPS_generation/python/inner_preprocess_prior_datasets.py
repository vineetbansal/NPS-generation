"""
Create input files for the cross-validation experiment by splitting into
cross-validation folds.
"""

import argparse
import itertools
import numpy as np
import os
import random
from rdkit import Chem
from selfies import encoder as selfies_encoder
from selfies.exceptions import EncoderError
from tqdm import tqdm

# import functions
from NPS_generation.python.functions import (
    read_smiles,
    write_smiles,
    clean_mols,
    remove_salts_solvents,
    NeutraliseCharges,
)
from NPS_generation.python.datasets import Vocabulary, SelfiesVocabulary
from NPS_generation.python.util.SmilesEnumerator import SmilesEnumerator


def add_args(parser):
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--vocab_file", type=str)
    parser.add_argument("--representation", type=str)
    parser.add_argument("--k", type=int)
    parser.add_argument("--cv_fold", type=int)
    parser.add_argument("--sample_idx", type=int)
    parser.add_argument("--enum_factor", type=int)
    return parser


def preprocess_prior_datasets(
    input_file,
    train_file,
    test_file,
    vocab_file,
    representation,
    k,
    cv_fold,
    sample_idx,
    enum_factor,
):
    # check output directory exists
    output_dir = os.path.dirname(train_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # read SMILES
    print("reading input SMILES ...")
    input_smiles = read_smiles(input_file)

    # convert to molecules
    print("converting {} input SMILES to molecules ...".format(len(input_smiles)))
    mols = clean_mols(input_smiles)

    # remove salts/solvents
    print("cleaning {} molecules ...".format(len(mols)))
    mols = [remove_salts_solvents(mol, hac=3) if mol else None for mol in tqdm(mols)]
    # remove charges
    mols = [NeutraliseCharges(mol) if mol else None for mol in tqdm(mols)]
    # remove molecules with invalid atoms
    # what unique atoms are present in any molecule?
    elements = [
        [atom.GetSymbol() for atom in mol.GetAtoms()] if mol else None for mol in mols
    ]
    # define valid symbols
    valid = set(["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"])
    for idx, atoms in enumerate(elements):
        if atoms is not None and len(set(atoms) - valid) > 0:
            mols[idx] = None
    # remove invalid molecules
    mols = [mol for mol in mols if mol is not None]

    # convert back to canonical SMILES
    print("converting {} molecules back to SMILES ...".format(len(mols)))
    canonical = [Chem.MolToSmiles(mol) for mol in tqdm(mols)]
    canonical = [sm for sm in canonical if sm != ""]
    canonical = list(dict.fromkeys(canonical))
    print("got {} unique canonical SMILES".format(len(canonical)))

    # split into folds
    random.seed(sample_idx)
    random.shuffle(canonical)
    fold_length = int(len(canonical) / k)
    folds = []
    for i in range(k - 1):
        folds += [canonical[i * fold_length : (i + 1) * fold_length]]

    folds += [canonical[(k - 1) * fold_length : len(canonical)]]

    # augment each fold
    if enum_factor > 0:
        # create enumerator
        sme = SmilesEnumerator(canonical=False, enum=True)
        for idx, fold in enumerate(folds):
            # enumerate potential SMILES
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

    # collapse fold into training/test datasets
    fold_idx = cv_fold - 1
    test = folds[fold_idx]
    train = folds[:fold_idx] + folds[fold_idx + 1 :]
    train = list(itertools.chain.from_iterable(train))

    # optionally, convert to SELFIES
    if representation == "SELFIES":
        print("converting SMILES strings to SELFIES ...")
        train_out = []
        for sm in train:
            try:
                sf = selfies_encoder(sm)
                train_out.append(sf)
            except EncoderError:
                pass

        test_out = []
        for sm in test:
            try:
                sf = selfies_encoder(sm)
                test_out.append(sf)
            except EncoderError:
                pass

        train = train_out
        test = test_out

    # then, write molecules
    write_smiles(train, train_file)
    write_smiles(test, test_file)

    # last, write vocabulary
    if representation == "SELFIES":
        vocabulary = SelfiesVocabulary(selfies=train)
    else:
        vocabulary = Vocabulary(smiles=train)

    print("vocabulary of {} characters:".format(len(vocabulary)))
    print(vocabulary.characters)
    tokens = vocabulary.characters
    with open(vocab_file, "w") as f:
        for token in tokens:
            _ = f.write(token + "\n")


def main(args):
    preprocess_prior_datasets(
        input_file=args.input_file,
        train_file=args.train_file,
        test_file=args.test_file,
        vocab_file=args.vocab_file,
        representation=args.representation,
        k=args.k,
        cv_fold=args.cv_fold,
        sample_idx=args.sample_idx,
        enum_factor=args.enum_factor,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
