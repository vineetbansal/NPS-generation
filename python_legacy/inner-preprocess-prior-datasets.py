"""
Create input files for the cross-validation experiment by splitting into 
cross-validation folds.
"""

import argparse
import itertools
import numpy as np
import os
import pandas as pd
import random
import sys
from itertools import chain
from rdkit import Chem
from selfies import encoder as selfies_encoder
from selfies.exceptions import EncoderError
from tqdm import tqdm


# import functions
from functions import read_smiles, write_smiles, clean_mols, \
    remove_salts_solvents, NeutraliseCharges
from datasets import Vocabulary, SelfiesVocabulary
from util.SmilesEnumerator import SmilesEnumerator

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--train_file', type=str)
parser.add_argument('--test_file', type=str)
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--representation', type=str)
parser.add_argument('--k', type=int)
parser.add_argument('--cv_fold', type=int)
parser.add_argument('--sample_idx', type=int)
parser.add_argument('--enum_factor', type=int)
# parse all arguments
args = parser.parse_args()
print(args)

# check output directory exists
output_dir = os.path.dirname(args.train_file)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# read SMILES
print('reading input SMILES ...')
input_smiles = read_smiles(args.input_file)

# convert to molecules
print('converting {} input SMILES to molecules ...'.format(len(input_smiles)))
mols = clean_mols(input_smiles)

# remove salts/solvents 
print('cleaning {} molecules ...'.format(len(mols)))
mols = [remove_salts_solvents(mol, hac=3) if mol else None for mol in \
        tqdm(mols)]
# remove charges
mols = [NeutraliseCharges(mol) if mol else None for mol in tqdm(mols)]
# remove molecules with invalid atoms 
## what unique atoms are present in any molecule? 
elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] if mol else None \
             for mol in mols]
counts = np.unique(list(chain(*[element for element in elements if element])),
                   return_counts=True)
## define valid symbols
valid = set(['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S'])
for idx, atoms in enumerate(elements):
    if atoms is not None and len(set(atoms) - valid) > 0:
        mols[idx] = None
# remove invalid molecules
mols = [mol for mol in mols if mol is not None]

# convert back to canonical SMILES
print('converting {} molecules back to SMILES ...'.format(len(mols)))
canonical = [Chem.MolToSmiles(mol) for mol in tqdm(mols)]
canonical = [sm for sm in canonical if sm != ""]
canonical = list(dict.fromkeys(canonical))
print('got {} unique canonical SMILES'.format(len(canonical)))

# split into folds
random.seed(args.sample_idx)
random.shuffle(canonical)
fold_length = int(len(canonical) / args.k)
folds = []
for i in range(args.k - 1):
    folds += [canonical[i * fold_length:(i+1) * fold_length]]

folds += [canonical[(args.k - 1) * fold_length:len(canonical)]]

# augment each fold
if args.enum_factor > 0:
    # create enumerator
    sme = SmilesEnumerator(canonical=False, enum=True)
    for idx, fold in enumerate(folds):
        # enumerate potential SMILES
        enum = []
        max_tries = 200 ## randomized SMILES to generate for each input structure
        for sm_idx, sm in enumerate(tqdm(fold)):
            tries = []
            for try_idx in range(max_tries):
                try:
                    this_try = sme.randomize_smiles(sm)
                    tries.append(this_try)
                    tries = [rnd for rnd in np.unique(tries)]
                    if len(tries) > args.enum_factor:
                        tries = tries[:args.enum_factor]
                        break
                except AttributeError:
                    continue
            enum.extend(tries)
        folds[idx] = enum

# collapse fold into training/test datasets
fold_idx = args.cv_fold - 1
test = folds[fold_idx]
train = folds[:fold_idx] + folds[fold_idx+1:]
train = list(itertools.chain.from_iterable(train))

# optionally, convert to SELFIES
if args.representation == 'SELFIES':
    print('converting SMILES strings to SELFIES ...')
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
write_smiles(train, args.train_file)
write_smiles(test, args.test_file)

# last, write vocabulary
if args.representation == 'SELFIES':
    vocabulary = SelfiesVocabulary(selfies=train)
else:
    vocabulary = Vocabulary(smiles=train)

print("vocabulary of {} characters:".format(len(vocabulary)))
print(vocabulary.characters)
tokens = vocabulary.characters
with open(args.vocab_file, 'w') as f:
    for token in tokens:
        _ = f.write(token + '\n')
