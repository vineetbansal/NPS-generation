"""
Sample generated molecules from a trained chemical language model.
"""

import argparse
import os
import os.path
import torch
from tqdm import tqdm

from datasets import Vocabulary, SelfiesDataset
from models import RNN
from functions import read_smiles

parser = argparse.ArgumentParser()

## build the CLI
parser.add_argument('--database', type=str)
parser.add_argument('--representation', type=str)
parser.add_argument('--rnn_type', type=str)
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--n_layers', type=int)
parser.add_argument('--dropout', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--sample_mols', type=int)
parser.add_argument('--input_file', type=str, default=None)
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--time_file', type=str)


# parse all arguments
args = parser.parse_args()
print(args)

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

# detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda: {}'.format(torch.cuda.is_available()))

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


# wipe the file before writing
open(args.output_file, 'w').close()

# sample a set of SMILES from the final, trained model
batch_size = 32
sampled_count = 0
with tqdm(total=args.sample_mols) as pbar:
    while sampled_count < args.sample_mols:
        sampled_smiles, losses = model.sample(batch_size, return_losses=True)
        # increment counter
        sampled_count += batch_size
        # write sampled SMILES
        with open(args.output_file, 'a+') as f:
            for loss, sm in zip(losses, sampled_smiles):
                f.write(str(round(loss, 4)) + ',' + sm + '\n')
        pbar.update(batch_size)
