import argparse
import numpy as np
import os
import pandas as pd
import sys
from itertools import chain
from rdkit import Chem
from selfies import encoder as selfies_encoder
from selfies.exceptions import EncoderError
from tqdm import tqdm

# set working directory
git_dir = os.path.expanduser("~/git/invalid-smiles-analysis")
python_dir = git_dir + "/python"
os.chdir(python_dir)
sys.path.append(python_dir)

# import functions
from functions import write_smiles, clean_mols, remove_salts_solvents, \
    NeutraliseCharges
from datasets import Vocabulary, SelfiesVocabulary
from util.SmilesEnumerator import SmilesEnumerator

### dynamically build CLI
parser = argparse.ArgumentParser()
## build the CLI
grid_file = git_dir + '/sh/grids/preprocess-gdb13-sample.txt'
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

# read SMILES
print('reading input SMILES ...')
sample = pd.read_csv(args.input_file, sep='\t', names=['smiles', 'freq', 'll'])
input_smiles = sample['smiles'].tolist()

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

# augmentation
smiles = canonical
if args.enum_factor > 0:
    # create enumerator
    sme = SmilesEnumerator(canonical=False, enum=True)
    enum = []
    max_tries = 200 ## randomized SMILES to generate for each input structure
    for sm_idx, sm in enumerate(tqdm(smiles)):
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
    smiles = enum

# optionally, convert to SELFIES
if args.representation == 'SELFIES':
    print('converting SMILES strings to SELFIES ...')
    sf_out = []
    for sm in smiles:
            try:
                sf = selfies_encoder(sm)
                sf_out.append(sf)
            except EncoderError:
                pass
    
    smiles = sf_out
            
# then, write molecules
write_smiles(smiles, args.output_file)

# last, write vocabulary
if args.representation == 'SELFIES':
    vocabulary = SelfiesVocabulary(selfies=smiles)
else:
    vocabulary = Vocabulary(smiles=smiles)

print("vocabulary of {} characters:".format(len(vocabulary)))
print(vocabulary.characters)
tokens = vocabulary.characters
with open(args.vocab_file, 'w') as f:
    for token in tokens:
        _ = f.write(token + '\n')
