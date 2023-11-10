"""
Create input files to train a chemical language model. 
"""

import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
from selfies import encoder as selfies_encoder
from selfies.exceptions import EncoderError
from tqdm import tqdm

# set working directory
git_dir = os.path.expanduser("~/git/invalid-smiles-analysis")
python_dir = git_dir + "/python"
os.chdir(python_dir)
sys.path.append(python_dir)

# import functions
from functions import read_smiles, clean_mols
from datasets import Vocabulary, SelfiesVocabulary
from util.SmilesEnumerator import SmilesEnumerator

### dynamically build CLI
parser = argparse.ArgumentParser()
## build the CLI
grid_file = git_dir + '/sh/grids/create-training-sets.txt'
grid = pd.read_csv(grid_file, sep='\t')
for arg_name in list(grid):
    param_name = '--' + arg_name
    param_dtype = str(grid[arg_name].dtype)
    # convert to pandas
    param_type = {'object': str,
                  'int64': int,
                  'float64': float,
                  'bool': str
                  }[param_dtype]
    parser.add_argument(param_name, type=param_type)

# parse all arguments
args = parser.parse_args()
print(args)

# check output directory exists
output_dir = os.path.dirname(args.output_file)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# set up input
input_dir = '/Genomics/skinniderlab/invalid-smiles-analysis'
input_file = input_dir + {'GDB13': 'GDB-13/gdb13_smiles.txt',
                          'ChEMBL': 'ChEMBL/chembl_28_smiles.txt'}[args.database]
# read SMILES
print('reading input SMILES ...')
input_smiles = read_smiles(input_file)

# convert to molecules and precalculate fingerprints for the entire input
print('converting {} input SMILES to molecules ...'.format(len(input_smiles)))
mols = clean_mols(input_smiles)
input_smiles = [input_smiles[idx] for idx, mol in enumerate(mols) if \
                mol is not None]
input_mols = [mol for mol in mols if mol is not None]
print('calculating fingerprints for {} valid molecules ...'.\
      format(len(input_mols)))
input_fps = [AllChem.GetMorganFingerprintAsBitVect(input_mol, 3, nBits=1024) \
             for input_mol in tqdm(input_mols)]

# shuffle SMILES and fingerprints
inputs = list(zip(input_smiles, input_fps))
random.seed(args.sample_idx)
random.shuffle(inputs)
input_smiles, input_fps = zip(*inputs)

# try to pick n molecules with minimum Tc to random seed molecule
max_tries = 100
success = False
for try_idx in range(max_tries):
    print('picking {} molecules with min_tc={}: try #{} of {} ...'.\
          format(args.n_molecules, args.min_tc, try_idx, max_tries))
    # shuffle SMILES and fingerprints again
    inputs = list(zip(input_smiles, input_fps))
    random.seed(try_idx)
    random.shuffle(inputs)
    input_smiles, input_fps = zip(*inputs)
    
    # pick our seed molecule at random
    target_smiles = input_smiles[0]
    target_fp = input_fps[0]
    
    # get Tanimoto coefficients
    tcs = [FingerprintSimilarity(input_fp, target_fp) for input_fp in \
           input_fps]
    # subset SMILES based on fingerprint similarity 
    subset_smiles = [input_smiles for input_smiles, tc in \
                     zip(input_smiles,tcs) if tc >= args.min_tc]
    
    # break if we have enough molecules
    if len(subset_smiles) >= args.n_molecules:
        subset_smiles = subset_smiles[:int(args.n_molecules)]
        success = True
        break

# if we failed to pick enough molecules, write an empty error file
if not success:
    error_file = os.path.splitext(args.output_file)[0] + ".err"
    with open(error_file, 'w') as empty_file: 
        pass 
else:
    # first, do SMILES enumeration on the training set
    print('doing SMILES enumeration on {} molecules ...'.\
          format(len(subset_smiles)))
    if args.enum_factor > 0:
        # create enumerator
        sme = SmilesEnumerator(canonical=False, enum=True)
        enum = []
        max_tries = 200
        for sm_idx, sm in enumerate(tqdm(subset_smiles)):
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
        subset_smiles = enum
    
    # then, optionally, convert to SELFIES
    if args.representation == 'SMILES':
        outputs = subset_smiles
    elif args.representation == 'SELFIES':
        print('converting {} SMILES strings to SELFIES ...'.\
          format(len(subset_smiles)))
        # encode to SELFIES
        outputs = []
        
        for sm in subset_smiles:
            try:
                sf = selfies_encoder(sm)
                outputs.append(sf)
            except EncoderError:
                pass
    
    # then, write molecules
    with open(args.output_file, 'w') as f:
        for idx, output in enumerate(outputs):
            if output is None:
                print("could not convert SMILES: {}".format(subset_smiles[idx]))
            else:
                _ = f.write(output + '\n')
    
    # last, write vocabulary
    ## no filtering of low-frequency tokens here
    if args.representation == 'SELFIES':
        vocabulary = SelfiesVocabulary(selfies=outputs)
    else:
        vocabulary = Vocabulary(smiles=outputs)
    
    print("vocabulary of {} characters:".format(len(vocabulary)))
    print(vocabulary.characters)
    tokens = vocabulary.characters
    with open(args.vocab_file, 'w') as f:
        for token in tokens:
            _ = f.write(token + '\n')
