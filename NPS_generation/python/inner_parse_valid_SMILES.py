"""
Parse SMILES sampled from a chemical language model, and flag them as invalid
(or, if invalid, flag the specific error category)
"""

import argparse
import os
import pandas as pd
import re
import sys
import time
from rdkit import Chem

from NPS_generation.python.datasets import Vocabulary, SelfiesVocabulary

# set working directory
git_dir = os.path.expanduser("~/git/invalid-smiles-analysis")
python_dir = git_dir + "/python"
os.chdir(python_dir)
sys.path.append(python_dir)

# dynamically build CLI
parser = argparse.ArgumentParser()
# build the CLI
grid_file = git_dir + "/sh/grids/parse-valid-SMILES.txt"
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

# create output directory if it does not exist
output_dir = os.path.dirname(args.output_file)
if not os.path.isdir(output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass


def cat_error(errors):
    """
    Error categorization from UnCorrupt SMILES paper.
    adapted from: https://github.com/LindeSchoenmaker/SMILES-corrector/blob/5fed15f2f4de5ac2bdc6cbaa986ae84a4b8bc6cd/analysis/figure.py#L62
    """
    error_list = []
    for error in errors:
        if error.find("Error") >= 1:
            # failed while parsing results in double mistake for one invalid smile
            # does however hold information on type of mistake
            # so not only pass those errors that do not contain while parsing
            if error.find("while parsing") == -1:
                error = re.split("Error:|for|while", error)
                if re.search("Failed", error[1]) is not None:
                    error[1] = "syntax error"
                elif re.search("ring closure", error[1]) is not None:
                    error[1] = "bond exists"
                # to merge parentheses errors together
                elif re.search("parentheses", error[1]) is not None:
                    error[1] = "parentheses error"
                error_list.append(error[1])
        elif error.find("valence") >= 1:
            error = "valence error"
            error_list.append(error)
        elif error.find("kekulize") >= 1:
            error = "aromaticity error"
            error_list.append(error)
        elif error.find("marked aromatic") >= 1:
            error = "aromaticity error"
            error_list.append(error)
        elif error.find("exists") >= 1:
            error = "bond exists"
            error_list.append(error)
        elif error.find("atoms") >= 1:
            # error = 'extra close parentheses'
            error = "parentheses error"
            error_list.append(error)
    return ";".join(error_list)


# start timer
start_time = time.time()

# open output file
output = open(args.output_file, "w")

# load vocabulary
if args.representation == "SELFIES":
    vocab = SelfiesVocabulary(vocab_file=args.vocab_file)
else:
    vocab = Vocabulary(vocab_file=args.vocab_file)

# read input line-by-line
line_idx = 0
f = open(args.input_file, "r")
# convoluted approach to save rdkit errors
# https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/CAKwxoo4HyYqsyw4RwZuruM1Cfi-nqZ-TuOSnBhb7yofSiATC%3Dw%40mail.gmail.com/#msg33261443
stderr_fileno = sys.stderr.fileno()
stderr_save = os.dup(stderr_fileno)
error_file = os.path.splitext(args.output_file)[0] + ".err"
stderr_fd = open(error_file, "w")
os.dup2(stderr_fd.fileno(), stderr_fileno)
for line in f:
    line_idx += 1

    # read SMILES
    smiles = line.split(",")[1].strip()
    # count number of tokens
    tokens = len(vocab.tokenize(smiles))
    # convert to molecule
    mol = Chem.MolFromSmiles(smiles)
    # get output
    if mol is None:
        # read from error
        err_conn = open(error_file, "r")
        msgs = err_conn.readlines()
        validity = ";".join(msgs).strip()
        errors = cat_error(msgs)
        print("invalid molecule: {}".format(validity))
        err_conn.close()
        # reset error
        stderr_fileno = sys.stderr.fileno()
        stderr_fd.close()
        stderr_fd = open(error_file, "w")
        os.dup2(stderr_fd.fileno(), stderr_fileno)
    else:
        validity = "valid"
        errors = ""

    # write to output
    _ = output.write(
        line.strip() + "," + str(tokens) + ',"' + validity + '",' + errors + "\n"
    )
    # abort if we have written enough
    if line_idx >= args.max_mols:
        break

# close output file
output.close()
# delete output file
os.remove(error_file)

# write the times
total_time = time.time() - start_time
timing_df = pd.DataFrame({"stage": ["total"], "time": [total_time]})
timing_df.to_csv(args.time_file, index=False)
