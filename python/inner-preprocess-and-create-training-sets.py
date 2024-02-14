"""
Create input files for the cross-validation experiment by splitting into
cross-validation folds and create input files to train a chemical language model.
"""

import argparse
import itertools
import numpy as np
import os
import random
from itertools import chain
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
from selfies import encoder as selfies_encoder
from selfies.exceptions import EncoderError
from tqdm import tqdm

# import functions
from functions import read_smiles, write_smiles, clean_mols, \
    remove_salts_solvents, NeutraliseCharges
from datasets import Vocabulary, SelfiesVocabulary
from util.SmilesEnumerator import SmilesEnumerator


def add_args(parser):
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--train_file_smi', type=str)
    parser.add_argument('--vocab_file_fps', type=str)
    parser.add_argument('--representation', type=str)
    parser.add_argument('--k', type=int)
    parser.add_argument('--cv_fold', type=int)
    parser.add_argument('--sample_idx', type=int)
    parser.add_argument('--enum_factor', type=int)
    parser.add_argument('--n_molecules', type=int)
    parser.add_argument('--min_tc', type=int)
    return parser


def preprocess_molecules(mols):
    print(f"cleaning {len(mols)} molecules ...")
    mols = [remove_salts_solvents(mol, hac=3) if mol else None for mol in tqdm(mols)]
    mols = [NeutraliseCharges(mol) if mol else None for mol in tqdm(mols)]

    # remove molecules with invalid atoms
    elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] if mol else None for mol in mols]
    counts = np.unique(list(chain(*[element for element in elements if element])), return_counts=True)

    valid_symbols = set(['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S'])
    for idx, atoms in enumerate(elements):
        if atoms is not None and len(set(atoms) - valid_symbols) > 0:
            mols[idx] = None
    mols = [mol for mol in mols if mol is not None]

    print(f"converting {len(mols)} molecules back to SMILES ...")
    canonical = [Chem.MolToSmiles(mol) for mol in tqdm(mols)]
    canonical = [sm for sm in canonical if sm != ""]
    canonical = list(dict.fromkeys(canonical))
    print(f"got {len(canonical)} unique canonical SMILES")

    return canonical


def create_training_sets(input_smiles, mols, sample_idx, n_molecules, min_tc):
    input_smiles = [input_smiles[idx] for idx, mol in enumerate(mols) if mol is not None]
    input_mols = [mol for mol in mols if mol is not None]
    print(f"calculating fingerprints for {len(input_mols)} valid molecules ...")
    input_fps = [AllChem.GetMorganFingerprintAsBitVect(input_mol, 3, nBits=1024) for input_mol in tqdm(input_mols)]

    # shuffle SMILES and fingerprints
    inputs = list(zip(input_smiles, input_fps))
    random.seed(sample_idx)
    random.shuffle(inputs)
    input_smiles, input_fps = zip(*inputs)

    # try to pick n molecules with minimum Tc to random seed molecule
    max_tries = 100
    success = False
    for try_idx in range(max_tries):
        print(f"picking {n_molecules} molecules with min_tc={min_tc} try #{try_idx} of {max_tries} ...")
        inputs = list(zip(input_smiles, input_fps))
        random.seed(try_idx)
        random.shuffle(inputs)
        input_smiles, input_fps = zip(*inputs)

        # pick our seed molecule at random
        target_smiles = input_smiles[0]
        target_fp = input_fps[0]

        tcs = [FingerprintSimilarity(input_fp, target_fp) for input_fp in input_fps]
        # subset SMILES based on fingerprint similarity
        subset_smiles = [input_smiles for input_smiles, tc in zip(input_smiles, tcs) if tc >= min_tc]

        # break if we have enough molecules
        if len(subset_smiles) >= n_molecules:
            subset_smiles = subset_smiles[:int(n_molecules)]
            success = True
            break

    return subset_smiles, success


def augment_smiles(enum_factor, folds):
    if enum_factor > 0:
        sme = SmilesEnumerator(canonical=False, enum=True)
        for idx, fold in enumerate(folds):
            enum = []
            max_tries = 200  ## randomized SMILES to generate for each input structure
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
    return folds


def convert_to_selfies(smiles):
    print('converting {} SMILES strings to SELFIES ...'.format(len(smiles)))
    selfies = []

    for sm in smiles:
        try:
            sf = selfies_encoder(sm)
            selfies.append(sf)
        except EncoderError:
            pass
    return selfies


def write_to_file(representation, output_molecules, output_file):
    if representation == 'SELFIES':
        output_molecules = convert_to_selfies(output_molecules)

    write_smiles(output_molecules, output_file)


def write_vocabulary(training_set, vocab_file, representation):
    if representation == 'SELFIES':
        training_set = convert_to_selfies(training_set)
        vocabulary = SelfiesVocabulary(selfies=training_set)
    else:
        vocabulary = Vocabulary(smiles=training_set)

    print("vocabulary of {} characters:".format(len(vocabulary)))
    print(vocabulary.characters)
    tokens = vocabulary.characters
    with open(vocab_file, 'w') as f:
        for token in tokens:
            _ = f.write(token + '\n')


def preprocess_and_create_training_sets(input_file, train_file, test_file, vocab_file, train_file_smi, vocab_file_fps,
                                        representation, k, cv_fold, sample_idx, enum_factor, n_molecules, min_tc):
    output_dir = os.path.dirname(train_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('reading input SMILES ...')
    input_smiles = read_smiles(input_file)

    print(f"converting {len(input_smiles)} input SMILES to molecules ...")
    mols = clean_mols(input_smiles)

    canonical = preprocess_molecules(mols)
    # split into folds
    random.seed(sample_idx)
    random.shuffle(canonical)
    fold_length = int(len(canonical) / k)
    folds = []
    for i in range(k - 1):
        folds += [canonical[i * fold_length:(i + 1) * fold_length]]

    folds += [canonical[(k - 1) * fold_length:len(canonical)]]
    folds = augment_smiles(enum_factor=enum_factor, folds=folds)

    # Collapse fold into training/test datasets
    fold_idx = cv_fold - 1
    test = folds[fold_idx]
    train = folds[:fold_idx] + folds[fold_idx + 1:]
    train = list(itertools.chain.from_iterable(train))

    subset_smiles, success = create_training_sets(input_smiles, mols, sample_idx, n_molecules, min_tc)

    # if we failed to pick enough molecules, write an empty error file
    if not success:
        error_file = os.path.splitext(train_file_smi)[0] + ".err"
        with open(error_file, 'w') as empty_file:
            pass
    else:
        print(f"doing SMILES enumeration on {len(subset_smiles)} molecules ...")
        subset_smiles = augment_smiles(enum_factor=enum_factor, folds=subset_smiles)

        # Write the molecules to a file
        write_to_file(representation, subset_smiles, train_file_smi)
        write_vocabulary(subset_smiles, vocab_file_fps, representation)

    write_to_file(representation, train, train_file)
    write_to_file(representation, test, test_file)
    write_vocabulary(train, vocab_file, representation)


def main(args):
    preprocess_and_create_training_sets(
        input_file=args.input_file,
        train_file=args.train_file,
        test_file=args.test_file,
        vocab_file=args.vocab_file,
        train_file_smi=args.train_file_smi,
        vocab_file_fps=args.vocab_file_fps,
        representation=args.representation,
        k=args.k,
        cv_fold=args.cv_fold,
        sample_idx=args.sample_idx,
        enum_factor=args.enum_factor,
        n_molecules=args.n_molecules,
        min_tc=args.min_tc,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
