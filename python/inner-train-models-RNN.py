"""
Train a RNN-based chemical language model on a dataset of known molecules.
"""

import argparse
import os
import pandas as pd
import sys
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# suppress Chem.MolFromSmiles error output
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

# import functions
from datasets import SmilesDataset, SelfiesDataset
from models import RNN
from functions import check_arg, read_smiles, write_smiles
from loggers import EarlyStopping, track_loss, print_update

parser = argparse.ArgumentParser()

parser.add_argument('--database', type=str)
parser.add_argument('--representation', type=str)
parser.add_argument('--enum_factor', type=int)
parser.add_argument('--n_molecules', type=int)
parser.add_argument('--min_tc', type=int)
parser.add_argument('--sample_idx', type=int)
parser.add_argument('--rnn_type', type=str)
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--n_layers', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--max_epochs', type=int)
parser.add_argument('--patience', type=int)
parser.add_argument('--log_every_steps', type=int)
parser.add_argument('--log_every_epochs', type=int)
parser.add_argument('--sample_mols', type=int)
parser.add_argument('--input_file', type=str)
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--smiles_file', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--loss_file', type=str)
parser.add_argument('--time_file', type=str)

# parse all arguments
args = parser.parse_args()
print(args)

# create output directory if it does not exist
output_dir = os.path.dirname(args.smiles_file)
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

# read training set
inputs = read_smiles(args.input_file)

# convert to dataset
if args.representation == 'SELFIES':
    dataset = SelfiesDataset(selfies=inputs, vocab_file=args.vocab_file)
else:
    dataset = SmilesDataset(smiles=inputs, vocab_file=args.vocab_file)

# create model
model = RNN(dataset.vocabulary,
            rnn_type=args.rnn_type,
            n_layers=args.n_layers,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout)
print(dataset.vocabulary.dictionary)

# set up data loader
loader = DataLoader(dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=dataset.collate)
# set up optimizer
optim = Adam(model.parameters(),
             betas=(0.9, 0.999),
             eps=1e-08,
             lr=args.learning_rate)
# set up early stopping
early_stop = EarlyStopping(patience=args.patience)

# another tick
train_start_time = time.time()

# iterate over epochs
counter = 0
for epoch in range(0, args.max_epochs):
    # iterate over batches
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        # increment counter
        counter += 1
        # abort?
        if check_arg(args, 'max_steps') and counter > args.max_steps:
            break

        # calculate loss
        loss = model.loss(batch)

        # zero gradients, calculate new gradients, and take a step
        optim.zero_grad()
        loss.backward()
        optim.step()

        # calculate validation loss
        validation = dataset.get_validation(args.batch_size)
        validation_loss = model.loss(validation).detach()

        # print update and write training schedule?
        if check_arg(args, 'log_every_steps'):
            if counter % args.log_every_steps == 0:
                track_loss(args.loss_file, epoch, counter,
                           loss.item(), validation_loss.item())
                print_update(model, epoch, batch_idx + 1, loss.item(),
                             validation_loss.item())

        # check early stopping
        early_stop(validation_loss.item(), model, args.model_file, counter)

        if early_stop.stop:
            break

    # print update and write training schedule?
    if check_arg(args, 'log_every_epochs'):
        track_loss(args.loss_file, epoch, counter,
                   loss.item(), validation_loss.item())
        print_update(model, epoch, 'NA', loss.item(), validation_loss.item())

    if early_stop.stop:
        break

# append information about minimum validation loss
if check_arg(args, 'log_every_epochs') or check_arg(args, 'log_every_steps'):
    row = pd.DataFrame({'epoch': [None],
                        'minibatch': [early_stop.step_at_best],
                        'outcome': ['training loss'],
                        'value': [early_stop.best_loss]})
    row.to_csv(args.loss_file, index=False, mode='a', header=False)

# another tick
sample_start_time = time.time()

# load the best model
model.load_state_dict(torch.load(args.model_file))
model.eval()

# sample a set of SMILES from the final, trained model
sampled_smiles = []
while len(sampled_smiles) < args.sample_mols:
    sampled_smiles.extend(model.sample(args.batch_size, return_smiles=True))

# write sampled SMILES
write_smiles(sampled_smiles, args.smiles_file)

# write the times
load_time = train_start_time - start_time
train_time = sample_start_time - train_start_time
sample_time = time.time() - sample_start_time
total_time = time.time() - start_time
timing_df = pd.DataFrame({'stage': ['load', 'train', 'sample', 'total'],
                          'time': [load_time, train_time, sample_time,
                                   total_time]})
timing_df.to_csv(args.time_file, index=False)
