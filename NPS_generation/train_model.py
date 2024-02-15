"""
Train a language model to generate SMILES.
"""

import argparse
import os
import numpy as np
import pandas as pd
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# import classes
from NPS_generation.models import RNN, OneHotRNN, EarlyStopping
from NPS_generation.datasets import SmilesDataset, SelfiesDataset, SmilesCollate
from NPS_generation.functions import (
    decrease_learning_rate,
    print_update,
    track_loss,
    sample_smiles,
    write_smiles,
)

# suppress Chem.MolFromSmiles error output
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")


def main(args_list=None):
    # CLI
    parser = argparse.ArgumentParser(
        description="Chemical structure language model interface"
    )
    # input file
    parser.add_argument(
        "--smiles_file", type=str, help="location of the SMILES file to train on"
    )
    parser.add_argument("--selfies", dest="selfies", action="store_true")
    parser.set_defaults(selfies=False)
    # output files
    parser.add_argument(
        "--output_dir", type=str, help="directory to save trained models to"
    )
    # RNN parameters
    parser.add_argument(
        "--rnn_type",
        type=str,
        choices=["RNN", "LSTM", "GRU"],
        default="GRU",
        help="type of language model to train",
    )
    parser.add_argument(
        "--embedding_size", type=int, default=128, help="size of vocabulary embedding"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="size of language model hidden layers",
    )
    parser.add_argument(
        "--n_layers", type=int, default=3, help="number of layers in language model"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        help="amount of dropout (0-1) to apply to model",
    )
    parser.add_argument(
        "--bidirectional",
        type=bool,
        default=False,
        help="for LSTMs only, train a bidirectional model",
    )
    parser.add_argument(
        "--nonlinearity",
        type=str,
        choices=["tanh", "relu"],
        default="tanh",
        help="for RNNs only, nonlinearity to use",
    )
    parser.add_argument(
        "--tie_weights",
        dest="tie_weights",
        help="require embedding/dense linear layers use the " + "same weights",
        action="store_true",
    )
    parser.set_defaults(tie_weights=False)
    # optimization parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="initial learning rate"
    )
    parser.add_argument(
        "--learning_rate_decay",
        default=None,  # type=float,
        help="amount (0-1) to decrease learning rate by every "
             + "fixed number of steps",
    )
    parser.add_argument(
        "--learning_rate_decay_steps",
        default=10000,
        type=int,
        help="# of steps between learning rate decrements",
    )
    parser.add_argument(
        "--gradient_clip",
        default=None,  # type=float,
        help="amount to which to clip the gradients",
    )
    # training schedule
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for random number generator"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="maximum number of epochs to train for",
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="patience for early stopping"
    )
    # sampling from trained models
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="index of the model being trained (zero-indexed)",
    )
    parser.add_argument(
        "--sample_every_epochs",
        type=int,
        help="if set, sample SMILES from the trained model" + "every n epochs",
    )
    parser.add_argument(
        "--sample_every_steps",
        type=int,
        help="if set, sample SMILES from the trained model" + "every n steps",
    )
    parser.add_argument(
        "--log_every_epochs",
        type=int,
        help="log training/validation losses every n epochs",
    )
    parser.add_argument(
        "--log_every_steps",
        type=int,
        help="log training/validation losses every n steps",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100000,
        help="size of each sample from the trained model",
    )
    # start with pretrained model
    parser.add_argument(
        "--pretrain_model",
        type=str,
        default=None,
        help="load parameters from a pretrained model",
    )
    # enforce a larger vocabulary
    parser.add_argument(
        "--vocab_file",
        type=str,
        default=None,
        help="file containing all tokens in vocabulary",
    )
    # for use in grid
    parser.add_argument("--stop_if_exists", dest="stop_if_exists", action="store_true")
    parser.set_defaults(stop_if_exists=False)

    # parse arguments
    args = parser.parse_args(args_list)

    # manually deal with gradient clipping
    try:
        args.gradient_clip = float(args.gradient_clip)
    except (ValueError, TypeError):
        args.gradient_clip = None

    # manually deal with learning rate decay
    try:
        args.learning_rate_decay = float(args.learning_rate_decay)
    except (ValueError, TypeError):
        args.learning_rate_decay = None

    # log args (make searching through logging directory easier)
    for arg in vars(args):
        print(arg, ": ", getattr(args, arg), "(", type(getattr(args, arg)), ")")

    # optionally stop if output file already exists
    if args.selfies:
        smiles_filename = "sample-" + str(args.sample_idx + 1) + "-SELFIES.smi"
    else:
        smiles_filename = "sample-" + str(args.sample_idx + 1) + "-SMILES.smi"
    smiles_file = os.path.join(args.output_dir, smiles_filename)
    if os.path.isfile(smiles_file) and args.stop_if_exists:
        print("output file " + smiles_file + " exists: stopping early")
        sys.exit()

    # make output directories
    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except FileExistsError:
            pass

    # seed all RNGs
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        print("using cuda")
        torch.cuda.manual_seed_all(args.seed)

    # set up dataset
    if args.selfies:
        dataset = SelfiesDataset(selfies_file=args.smiles_file)
    else:
        dataset = SmilesDataset(
            smiles_file=args.smiles_file, vocab_file=args.vocab_file
        )

    # set up batching
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=SmilesCollate(dataset.vocabulary),
    )

    # set up model
    if args.embedding_size > 0:
        model = RNN(
            vocabulary=dataset.vocabulary,
            rnn_type=args.rnn_type,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            tie_weights=args.tie_weights,
            nonlinearity=args.nonlinearity,
        )
    else:
        # no embedding layer (one-hot encoding)
        model = OneHotRNN(
            vocabulary=dataset.vocabulary,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            nonlinearity=args.nonlinearity,
        )

    # optionally, load model parameters from file
    if args.pretrain_model is not None:
        model.load_state_dict(torch.load(args.pretrain_model))

    # set up optimizer
    optimizer = optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),  # default
        eps=1e-08,  # default
        lr=args.learning_rate,
    )

    # set up early stopping
    early_stop = EarlyStopping(patience=args.patience)

    # set up training schedule file
    sched_filename = "training_schedule-" + str(args.sample_idx + 1) + ".csv"
    sched_file = os.path.join(args.output_dir, sched_filename)

    # iterate over epochs
    counter = 0
    for epoch in range(args.max_epochs):
        # iterate over batches
        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
            batch, lengths = batch

            # increment counter
            counter += 1

            # calculate loss
            log_p = model.loss(batch, lengths)
            loss = log_p.mean()

            # zero gradients, calculate new gradients, and take a step
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            if args.gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optimizer.step()

            # check learning rate decay
            if (
                    args.learning_rate_decay is not None
                    and counter % args.learning_rate_decay_steps == 0
            ):
                decrease_learning_rate(optimizer, multiplier=args.learning_rate_decay)

            # print update and write training schedule?
            if args.log_every_steps is not None:
                if counter % args.log_every_steps == 0:
                    print_update(
                        model,
                        dataset,
                        epoch,
                        batch_idx + 1,
                        loss.item(),
                        args.batch_size,
                        selfies=args.selfies,
                    )
                    track_loss(
                        sched_file,
                        model,
                        dataset,
                        epoch,
                        counter,
                        loss.item(),
                        args.batch_size,
                    )

            # save SMILES?
            if args.sample_every_steps is not None:
                if counter % args.sample_every_steps == 0:
                    sample_smiles(
                        args.output_dir,
                        args.sample_idx,
                        model,
                        args.sample_size,
                        epoch,
                        counter,
                    )

            # calculate validation loss
            validation, lengths = dataset.get_validation(args.batch_size)
            validation_loss = model.loss(validation, lengths).mean().detach()
            # check early stopping
            model_filename = "model-" + str(args.sample_idx + 1) + ".pt"
            model_file = os.path.join(args.output_dir, model_filename)
            early_stop(validation_loss.item(), model, model_file, counter)

            if early_stop.stop:
                break

        # print update and write training schedule?
        if args.log_every_epochs is not None:
            print_update(model, dataset, epoch, "NA", loss.item(), args.batch_size)
            track_loss(
                sched_file, model, dataset, epoch, counter, loss.item(), args.batch_size
            )

        # save SMILES?
        if args.sample_every_epochs is not None:
            sample_smiles(
                args.output_dir,
                args.sample_idx,
                model,
                args.sample_size,
                epoch,
                counter,
            )

        if early_stop.stop:
            break

    # append information about final training step
    if args.log_every_epochs is not None or args.log_every_steps is not None:
        sched = pd.DataFrame(
            {
                "epoch": [None],
                "step": [early_stop.step_at_best],
                "outcome": ["training loss"],
                "value": [early_stop.best_loss],
            }
        )
        sched.to_csv(sched_file, index=False, mode="a", header=False)

    # load the best model
    model.load_state_dict(torch.load(model_file))
    model.eval()  # enable evaluation modes

    # sample a set of SMILES from the final, trained model
    sampled_smiles = []
    while len(sampled_smiles) < args.sample_size:
        sampled_smiles.extend(model.sample(args.batch_size, return_smiles=True))

    # write sampled SMILES
    write_smiles(sampled_smiles, smiles_file)


if __name__ == "__main__":
    main()
