"""
Parse SELFIES sampled from a chemical language model _without_ valency
constraints, and write valid vs. invalid SELFIES to separate files.
"""

import argparse
import os
import pandas as pd
import re
import selfies as sf
import sys
import time
from rdkit import Chem

# set working directory
git_dir = os.path.expanduser("~/git/invalid-smiles-analysis")
python_dir = git_dir + "/python"
os.chdir(python_dir)
sys.path.append(python_dir)

from datasets import Vocabulary, SelfiesVocabulary

### dynamically build CLI
parser = argparse.ArgumentParser()
## build the CLI
grid_file = git_dir + '/sh/grids/parse-valid-SELFIES.txt'
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

# create output directory if it does not exist
output_dir = os.path.dirname(args.valid_file)
if not os.path.isdir(output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

# remove SELFIES constraints
if args.constraints == 'none':
    _NO_CONSTRAINTS = {
        "H": 999, "F": 999, "Cl": 999, "Br": 999, "I": 999,
        "B": 999, "B+1": 999, "B-1": 999,
        "O": 999, "O+1": 999, "O-1": 999,
        "N": 999, "N+1": 999, "N-1": 999,
        "C": 999, "C+1": 999, "C-1": 999,
        "P": 999, "P+1": 999, "P-1": 999,
        "S": 999, "S+1": 999, "S-1": 999,
        "?": 999
    }
    sf.set_semantic_constraints(_NO_CONSTRAINTS)
elif args.constraints == 'c5': 
    _TEXAS_CONSTRAINTS = {
        "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1,
        "B": 3, "B+1": 2, "B-1": 4,
        "O": 2, "O+1": 3, "O-1": 1,
        "N": 3, "N+1": 4, "N-1": 2,
        "C": 5, "C+1": 5, "C-1": 3,
        "P": 5, "P+1": 6, "P-1": 4,
        "S": 6, "S+1": 7, "S-1": 5,
        "?": 8
    }
    sf.set_semantic_constraints(_TEXAS_CONSTRAINTS)

def cat_error(errors):
    """
    Error categorization from UnCorrupt SMILES paper.
    adapted from: https://github.com/LindeSchoenmaker/SMILES-corrector/blob/5fed15f2f4de5ac2bdc6cbaa986ae84a4b8bc6cd/analysis/figure.py#L62
    """
    error_list = []
    for error in errors:
        if error.find("Error") >= 1:
            #failed while parsing results in double mistake for one invalid smile
            #does however hold information on type of mistake
            #so not only pass those errors that do not contain while parsing
            if error.find("while parsing") == -1:
                error = re.split('Error:|for|while', error)
                if re.search('Failed', error[1]) != None:
                    error[1] = 'syntax error'
                elif re.search('ring closure', error[1]) != None:
                    error[1] = 'bond exists'
                # to merge parentheses errors together
                elif re.search('parentheses', error[1]) != None:
                    error[1] = 'parentheses error'
                error_list.append(error[1])
        elif error.find("valence") >= 1:
            error = 'valence error'
            error_list.append(error)
        elif error.find("kekulize") >= 1:
            error = 'aromaticity error'
            error_list.append(error)
        elif error.find("marked aromatic") >= 1:
            error = 'aromaticity error'
            error_list.append(error)
        elif error.find("exists") >= 1:
            error = 'bond exists'
            error_list.append(error)
        elif error.find("atoms") >= 1:
            #error = 'extra close parentheses'
            error = 'parentheses error'
            error_list.append(error)
    return ";".join(error_list)

# start timer
start_time = time.time()

# load vocabulary
if args.representation == 'SELFIES':
    vocab = SelfiesVocabulary(vocab_file=args.vocab_file)
else:
    vocab = Vocabulary(vocab_file=args.vocab_file)

# open invalid and valid files
valid_output = open(args.valid_file, 'w')
invalid_output = open(args.invalid_file, 'w')
# write headers
header = "loss,smiles,tokens,valid,errors,selfies\n"
valid_output.write(header)
invalid_output.write(header)

# read input line-by-line
line_idx = 0
f = open(args.input_file, 'r')
# convoluted approach to save rdkit errors
## https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/CAKwxoo4HyYqsyw4RwZuruM1Cfi-nqZ-TuOSnBhb7yofSiATC%3Dw%40mail.gmail.com/#msg33261443
stderr_fileno = sys.stderr.fileno()
stderr_save = os.dup(stderr_fileno)
error_file = os.path.splitext(args.valid_file)[0] + ".err"
stderr_fd = open(error_file, 'w')
os.dup2(stderr_fd.fileno(), stderr_fileno)
for line in f:
    line_idx += 1
    
    # read SELFIES
    loss = line.split(',')[0].strip()
    selfies = line.split(',')[1].strip()
    # count number of tokens
    selfies = selfies.replace('<PAD>', '[nop]')
    try:
        tokens = len(vocab.tokenize(selfies))
    except ValueError:
        tokens = 'NA'
    # decode SELFIES without any constraints 
    smiles = sf.decoder(selfies)
    # convert to molecule
    mol = Chem.MolFromSmiles(smiles)
    # get output
    if mol is None:
        # read from error
        err_conn = open(error_file, 'r')
        msgs = err_conn.readlines()
        validity = ";".join(msgs).strip()
        errors = cat_error(msgs)
        print('invalid molecule: {}'.format(validity))
        err_conn.close()
        # reset error
        stderr_fileno = sys.stderr.fileno()
        stderr_fd.close()
        stderr_fd = open(error_file, 'w')
        os.dup2(stderr_fd.fileno(), stderr_fileno)
        # write to invalid file
        _ = invalid_output.write(loss + ',' + smiles + ',' + str(tokens) + \
                                 ',"' + validity + '",' + errors + ',' + \
                                     selfies + '\n')
    else:
        validity = 'valid'
        errors = ''
        # write to valid file
        _ = valid_output.write(loss + ',' + smiles + ',' + str(tokens) + \
                               ',"' + validity + '",' + errors + ',' + \
                                   selfies + '\n')
    
    # abort if we have written enough
    if line_idx >= args.max_mols:
        break

# close output files
valid_output.close()
invalid_output.close()
# delete output file
os.remove(error_file)

# write the times
total_time = time.time() - start_time
timing_df = pd.DataFrame({'stage': ['total'], 'time': [total_time]})
timing_df.to_csv(args.time_file, index=False)
