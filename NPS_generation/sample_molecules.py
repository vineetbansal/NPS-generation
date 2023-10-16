"""
Sample generated molecules from a trained chemical language model.
"""

import argparse
import numpy as np
import os
import random
import torch

# set working directory
git_dir = os.path.expanduser("~/git/NPS_generation")
python_dir = git_dir + "/python"
os.chdir(python_dir)

# import classes and functions
from datasets import Vocabulary, SmilesDataset
from models import RNN, OneHotRNN

### CLI
parser = argparse.ArgumentParser(
        description='Sample from a trained chemical language model')
parser.add_argument('--model_file', type=str,
                    help='location of the SMILES file to train on')
parser.add_argument('--vocab_file', type=str,
                    help='vocabulary the model was trained on')
parser.add_argument('--smiles_file', type=str,
                    help='SMILES file the model was trained on')
parser.add_argument('--output_dir', type=str,
                    help='directory to save trained models to')
parser.add_argument('--mols_per_file', type=int, default=1e7,
                    help='number of sampled molecules per file')
parser.add_argument('--sample_idx', type=int, default=None,
                    help='index of the current sample')
# RNN parameters
parser.add_argument('--rnn_type', type=str, choices=['RNN', 'LSTM', 'GRU'],
                    default='GRU', help='type of language model to train')
parser.add_argument('--embedding_size', type=int, default=128,
                    help='size of vocabulary embedding')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='size of language model hidden layers')
parser.add_argument('--n_layers', type=int, default=3,
                    help='number of layers in language model')
parser.add_argument('--dropout', type=float, default=0,
                    help='amount of dropout (0-1) to apply to model')
parser.add_argument('--bidirectional', type=bool, default=False,
                    help='for LSTMs only, train a bidirectional model')
parser.add_argument('--nonlinearity', type=str, choices=['tanh', 'relu'],
                    default='tanh', help='for RNNs only, nonlinearity to use')
parser.add_argument('--tie_weights', dest='tie_weights',
                    help='require embedding/dense linear layers use the ' +\
                    'same weights',
                    action='store_true')
parser.set_defaults(tie_weights=False)
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

## seed all RNGs
torch.manual_seed(args.sample_idx)
random.seed(args.sample_idx)
np.random.seed(args.sample_idx)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.sample_idx)

# load vocabulary
if args.vocab_file is not None:
    vocab = Vocabulary(vocab_file=args.vocab_file)
elif args.smiles_file is not None:
    dataset = SmilesDataset(smiles_file=args.smiles_file, vocab_file=None)
    vocab = dataset.vocabulary

# set up model
if args.embedding_size > 0:
    model = RNN(vocabulary=vocab,
                rnn_type=args.rnn_type,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                n_layers=args.n_layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional,
                tie_weights=args.tie_weights,
                nonlinearity=args.nonlinearity)
else:
    # no embedding layer (one-hot encoding)
    model = OneHotRNN(vocabulary=vocab,
                      rnn_type=args.rnn_type,
                      hidden_size=args.hidden_size,
                      n_layers=args.n_layers,
                      dropout=args.dropout,
                      bidirectional=args.bidirectional,
                      nonlinearity=args.nonlinearity)

# load the best model
model.load_state_dict(torch.load(args.model_file))
model.eval() ## enable evaluation modes

# set up output filename
if args.sample_idx is not None:
    output_filename = 'sampled-SMILES-{}.smi'.format(args.sample_idx)
else:
    output_filename = 'sampled-SMILES.smi'

output_file = os.path.join(args.output_dir, output_filename)

# sample a set of SMILES from the final, trained model
sampled_count = 0
batch_size = 512
while sampled_count < args.mols_per_file:
    sampled_smiles, NLLs = model.sample(batch_size, return_smiles=True,
                                        return_nll=True)
    # write sampled SMILES
    with open(output_file, 'a+') as f:
        for sm, nll in zip(sampled_smiles, NLLs):
            _ = f.write(sm + '\t' + str(round(nll.detach().item(), 6)) + '\n')

    sampled_count += batch_size
