"""
Calculate the Tanimoto coefficient between random pairs of SMILES in a
database.
"""

import argparse
import os
import pandas as pd
import random
from rdkit.DataStructs import FingerprintSimilarity

# import functions
from functions import clean_mols, get_ecfp6_fingerprints, read_smiles

# suppress rdkit errors
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")

# set working directory
git_dir = os.path.expanduser("~/git/invalid-smiles-analysis")
python_dir = git_dir + "/python"
os.chdir(python_dir)

# dynamically build CLI
parser = argparse.ArgumentParser()
# build the CLI
grid_file = git_dir + "/sh/grids/write-pairwise-Tc-distribution.txt"
grid = pd.read_csv(grid_file, sep="\t")
for arg_name in list(grid):
    param_name = "--" + arg_name
    param_dtype = str(grid[arg_name].dtype)
    # convert to pandas
    param_type = {"object": str, "int64": int, "float64": float, "bool": str}[
        param_dtype
    ]
    parser.add_argument(param_name, type=param_type)

# parse all arguments
args = parser.parse_args()
print(args)

output_dir = os.path.dirname(args.output_file)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# read training and test sets
train_smiles = read_smiles(args.train_file)
test_smiles = read_smiles(args.test_file)
smiles = train_smiles
smiles.extend(test_smiles)

# convert to molecules
mols = clean_mols(smiles)
# get fingerprints
fps = get_ecfp6_fingerprints(mols)

# get pairs
idxs = range(len(mols))
pairs = [random.sample(idxs, 2) for i in range(int(args.n))]
idx1 = [i[0] for i in pairs]
idx2 = [i[1] for i in pairs]
smiles1 = [smiles[idx] for idx in idx1]
smiles2 = [smiles[idx] for idx in idx2]
fps1 = [fps[idx] for idx in idx1]
fps2 = [fps[idx] for idx in idx2]

# calculate Tc
tcs = [FingerprintSimilarity(fps1[idx], fps2[idx]) for idx in range(int(args.n))]

# create output
tc_df = pd.DataFrame(
    {"smiles1": smiles1, "smiles2": smiles2, "tc": tcs}
).drop_duplicates()

# write to output files
tc_df.to_csv(args.output_file, index=False, compression="gzip")
