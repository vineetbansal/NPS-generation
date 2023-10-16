"""
Take an input SMILES file, and augment it by some fixed factor via SMILES
enumeration.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# set working directory
git_dir = os.path.expanduser("~/documents/NPS_generation")
python_dir = git_dir + "/NPS_generation"
os.chdir(python_dir)

# import SmilesEnumerator
from util.SmilesEnumerator import SmilesEnumerator
from functions import read_smiles, write_smiles

### CLI
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--enum_factor', type=int,
                    help='factor to augment the dataset by')
args = parser.parse_args()

# check output directory exists
output_dir = os.path.dirname(args.output_file)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# read SMILES
smiles = read_smiles(args.input_file)
# convert to numpy array
smiles = np.asarray(smiles)

# create enumerator
sme = SmilesEnumerator(canonical=False, enum=True)

# also store and write information about enumerated library size
summary = pd.DataFrame()

# enumerate potential SMILES
enum = []
max_tries = 200 ## randomized SMILES to generate for each input structure
for sm_idx, sm in enumerate(tqdm(smiles)):
    tries = []
    for try_idx in range(max_tries):
        this_try = sme.randomize_smiles(sm)
        tries.append(this_try)
        tries = [rnd for rnd in np.unique(tries)]
        if len(tries) > args.enum_factor:
            tries = tries[:args.enum_factor]
            break
    enum.extend(tries)

# write to line-delimited file
write_smiles(enum, args.output_file)
print("wrote " + str(len(enum)) + " SMILES to output file: " + \
      args.output_file)
