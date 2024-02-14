"""
Train a RNN-based chemical language model on a dataset of known molecules.
"""

import argparse
import os
import os.path
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# suppress Chem.MolFromSmiles error output
from rdkit import rdBase

rdBase.DisableLog('rdApp.error')

from datasets import SmilesDataset, SelfiesDataset
from models import RNN
from functions import check_arg, read_smiles, write_smiles
from loggers import EarlyStopping, track_loss, print_update
from NPS_generation.functions import set_seed, seed_type


def add_args(parser):
    parser.add_argument('--database', type=str)
    parser.add_argument('--representation', type=str)
    parser.add_argument('--seed', type=seed_type, default=None, nargs="?", help="Random seed")
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
    parser.add_argument('--smiles_file', type=str, default=None)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--loss_file', type=str)

    return parser


def train_models_RNN(database, representation, seed, rnn_type, embedding_size, hidden_size, n_layers, dropout,
                     batch_size, learning_rate, max_epochs, patience, log_every_steps, log_every_epochs,
                     sample_mols, input_file, vocab_file, smiles_file, model_file, loss_file):

    set_seed(seed)

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(os.path.dirname(loss_file), exist_ok=True)

    # detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda: {}'.format(torch.cuda.is_available()))

    # read training set
    inputs = read_smiles(input_file)

    # convert to dataset
    if representation == 'SELFIES':
        dataset = SelfiesDataset(selfies=inputs, vocab_file=vocab_file)
    else:
        dataset = SmilesDataset(smiles=inputs, vocab_file=vocab_file)

    # create model
    model = RNN(dataset.vocabulary,
                rnn_type=rnn_type,
                n_layers=n_layers,
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                dropout=dropout)
    print(dataset.vocabulary.dictionary)

    # set up data loader
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=dataset.collate)
    # set up optimizer
    optim = Adam(model.parameters(),
                 betas=(0.9, 0.999),
                 eps=1e-08,
                 lr=learning_rate)
    # set up early stopping
    early_stop = EarlyStopping(patience=patience)

    # iterate over epochs
    counter = 0
    for epoch in range(0, max_epochs):
        # iterate over batches
        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
            # increment counter
            counter += 1
            # abort?
            # TODO: figure out how to include max_steps
            # if max_steps and counter > max_steps:
            #     break

            # calculate loss
            loss = model.loss(batch)

            # zero gradients, calculate new gradients, and take a step
            optim.zero_grad()
            loss.backward()
            optim.step()

            # calculate validation loss
            validation = dataset.get_validation(batch_size)
            validation_loss = model.loss(validation).detach()

            # print update and write training schedule?
            if check_arg(args, 'log_every_steps'):
                if counter % log_every_steps == 0:
                    track_loss(loss_file, epoch, counter,
                               loss.item(), validation_loss.item())
                    print_update(model, epoch, batch_idx + 1, loss.item(),
                                 validation_loss.item())

            # check early stopping
            early_stop(validation_loss.item(), model, model_file, counter)

            if early_stop.stop:
                break

        # print update and write training schedule?
        if log_every_epochs:
            track_loss(loss_file, epoch, counter,
                       loss.item(), validation_loss.item())
            print_update(model, epoch, 'NA', loss.item(), validation_loss.item())

        if early_stop.stop:
            break

    # append information about minimum validation loss
    if log_every_epochs or log_every_steps:
        row = pd.DataFrame({'epoch': [None],
                            'minibatch': [early_stop.step_at_best],
                            'outcome': ['training loss'],
                            'value': [early_stop.best_loss]})
        row.to_csv(loss_file, index=False, mode='a', header=False)

    # load the best model
    model.load_state_dict(torch.load(model_file))
    model.eval()

    if smiles_file is not None:
        # sample a set of SMILES from the final, trained model
        sampled_smiles = []
        with tqdm(total=sample_mols) as pbar:
            while len(sampled_smiles) < sample_mols:
                new_smiles = model.sample(batch_size, return_smiles=True)
                sampled_smiles.extend(new_smiles)
                pbar.update(len(new_smiles))

        # write sampled SMILES
        write_smiles(sampled_smiles, smiles_file)


def main(args):
    train_models_RNN(
        database=args.database,
        representation=args.representation,
        seed=args.seed,
        rnn_type=args.rnn_type,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_every_steps=args.log_every_steps,
        log_every_epochs=args.log_every_epochs,
        sample_mols=args.sample_mols,
        input_file=args.input_file,
        vocab_file=args.vocab_file,
        smiles_file=args.smiles_file,
        model_file=args.model_file,
        loss_file=args.loss_file,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
