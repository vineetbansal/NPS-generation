"""
Sample generated molecules from a trained chemical language model.
"""

import argparse
import os
import pandas as pd
import sys
import time
import torch

# set working directory
git_dir = os.path.expanduser("~/git/invalid-smiles-analysis")
python_dir = git_dir + "/python"
os.chdir(python_dir)
sys.path.append(python_dir)

# import classes and functions
from datasets import Vocabulary, SelfiesDataset
from models import RNN
from functions import read_smiles

### dynamically build CLI
parser = argparse.ArgumentParser()
## build the CLI
grid_file = git_dir + '/sh/grids/sample-molecules-RNN.txt'
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
output_dir = os.path.dirname(args.output_file)
if not os.path.isdir(output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

# detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda: {}'.format(torch.cuda.is_available()))

# start the timer
start_time = time.time()

# load vocabulary
if args.representation == 'SELFIES':
    inputs = read_smiles(args.input_file)
    dataset = SelfiesDataset(selfies=inputs, vocab_file=args.vocab_file)
    vocab = dataset.vocabulary
else:
    vocab = Vocabulary(vocab_file=args.vocab_file)

# set up model
model = RNN(vocab,
            rnn_type=args.rnn_type,
            n_layers=args.n_layers,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout)
print(vocab.dictionary)

# load the trained model parameters
if torch.cuda.is_available():
    model.load_state_dict(torch.load(args.model_file))
else:
    model.load_state_dict(torch.load(args.model_file,
                                     map_location=torch.device('cpu')))

model.eval() ## enable evaluation mode

# another tick
sample_start_time = time.time()

# wipe the file before writing
open(args.output_file, 'w').close()

# sample a set of SMILES from the final, trained model
batch_size = 512
sampled_count = 0
while sampled_count < args.sample_mols:
    sampled_smiles, losses = model.sample(batch_size, return_losses=True)
    # increment counter
    sampled_count += batch_size
    # write sampled SMILES
    with open(args.output_file, 'a+') as f:
        for loss, sm in zip(losses, sampled_smiles):
                _ = f.write(str(round(loss, 4)) + ',' + sm + '\n')

# write the times
load_time = sample_start_time - start_time
sample_time = time.time() - sample_start_time
total_time = time.time() - start_time
timing_df = pd.DataFrame({'stage': ['load', 'sample', 'total'],
                          'time': [load_time, sample_time, total_time]})
timing_df.to_csv(args.time_file, index=False)
