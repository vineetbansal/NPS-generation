"""
For a large set of molecules sampled from a model, tabulate unique canonical
SMILES and record the following properties:
- SMILES
- exact mass
- molecular formula
- sampling frequency
"""

import argparse
import os
import pandas as pd
import selfies as sf
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# import functions
from functions import clean_mol, read_smiles

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
grid_file = git_dir + "/sh/grids/tabulate-molecules-break-SELFIES.txt"
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

# tick
start_time = time.time()

# read training SMILES
train_smiles = read_smiles(args.train_file)

# set up temporary output
filename, ext = os.path.splitext(args.output_file)
tmp_file = filename + ".temp"
# remove file if it exists
if os.path.exists(tmp_file):
    os.remove(tmp_file)

f2 = open(tmp_file, "a+")

# remove SELFIES constraints
if args.constraints == "none":
    _NO_CONSTRAINTS = {
        "H": 999,
        "F": 999,
        "Cl": 999,
        "Br": 999,
        "I": 999,
        "B": 999,
        "B+1": 999,
        "B-1": 999,
        "O": 999,
        "O+1": 999,
        "O-1": 999,
        "N": 999,
        "N+1": 999,
        "N-1": 999,
        "C": 999,
        "C+1": 999,
        "C-1": 999,
        "P": 999,
        "P+1": 999,
        "P-1": 999,
        "S": 999,
        "S+1": 999,
        "S-1": 999,
        "?": 999,
    }
    sf.set_semantic_constraints(_NO_CONSTRAINTS)
elif args.constraints == "c5":
    _TEXAS_CONSTRAINTS = {
        "H": 1,
        "F": 1,
        "Cl": 1,
        "Br": 1,
        "I": 1,
        "B": 3,
        "B+1": 2,
        "B-1": 4,
        "O": 2,
        "O+1": 3,
        "O-1": 1,
        "N": 3,
        "N+1": 4,
        "N-1": 2,
        "C": 5,
        "C+1": 5,
        "C-1": 3,
        "P": 5,
        "P+1": 6,
        "P-1": 4,
        "S": 6,
        "S+1": 7,
        "S-1": 5,
        "?": 8,
    }
    sf.set_semantic_constraints(_TEXAS_CONSTRAINTS)

# read SMILES line-by-line, and calculate properties for real molecules
with open(args.input_file) as f1:
    for line_idx, line in enumerate(f1):
        # split line
        split = line.strip().split(",")
        if len(split) == 2:
            mass, smiles = split[0], split[1]
        else:
            smiles = split[0]

        # try to parse the molecule
        try:
            mol = clean_mol(smiles, representation=args.representation)

            # calculate exact mass
            exact_mass = Descriptors.ExactMolWt(mol)
            # round to 6 decimal places
            mass = round(exact_mass, 6)

            # calculate molecular formula
            formula = rdMolDescriptors.CalcMolFormula(mol)

            # roundtrip to get canonical smiles
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

            # optionally, skip
            if canonical_smiles not in train_smiles:
                # append to file
                row = "\t".join([canonical_smiles, str(mass), formula])
                _ = f2.write(row + "\n")
                f2.flush()
        except ValueError:
            next

# read temporary output, and tabulate frequencies
# NOTE: group by canonical SMILES and pick the best log-likelihood
df = pd.read_csv(tmp_file, sep="\t", header=None, names=["smiles", "mass", "formula"])
# calculate frequency of each canonical SMILES
freqs = (
    df.groupby(["smiles", "mass", "formula"])
    .size()
    .to_frame("size")
    .reset_index()
    .sort_values("size", ascending=False)
)
# write
freqs.to_csv(args.output_file, index=False)

# remove the tmp_file
os.remove(tmp_file)

# write time
total_time = time.time() - start_time
timing_df = pd.DataFrame({"stage": ["total"], "time": [total_time]})
timing_df.to_csv(args.time_file, index=False)
