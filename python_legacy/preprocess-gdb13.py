"""
Preprocess the entire GDB-13 dataset.
"""

import argparse
import os
import sys
from rdkit import Chem

# set working directory
git_dir = os.path.expanduser("~/git/invalid-smiles-analysis")
python_dir = git_dir + "/python"
os.chdir(python_dir)
sys.path.append(python_dir)

# import functions
from functions import clean_mol, remove_salts_solvents, NeutraliseCharges

### dynamically build CLI
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
args = parser.parse_args()
print(args)

# check output directory exists
output_dir = os.path.dirname(args.output_file)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# set up output
if os.path.exists(args.output_file):
  os.remove(args.output_file)

f_out = open(args.output_file, 'a+')

# read SMILES
print('reading input SMILES ...')
with open(args.input_file) as f:
    for line_idx, line in enumerate(f):
        # split line
        split = line.strip().split('\t')
        smiles = split[0]
        
        # convert to molecule
        mol = clean_mol(smiles)
        if mol is None:
            next
        # remove salts/solvents
        mol = remove_salts_solvents(mol, hac=3)
        if mol is None: 
            next
        # remove charges
        mol = NeutraliseCharges(mol)
        if mol is None: 
            next
        # convert back to canonical SMILES
        canonical = Chem.MolToSmiles(mol)
        if canonical == "":
            next
        
        # append to file
        _ = f_out.write(canonical + '\n')
        f_out.flush()
