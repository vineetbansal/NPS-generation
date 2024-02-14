"""
Sample generated molecules from a trained chemical language model.
"""

import argparse
import os.path
import torch
from tqdm import tqdm

from datasets import Vocabulary, SelfiesDataset
from models import RNN
from functions import read_smiles
from NPS_generation.functions import set_seed, seed_type


def add_args(parser):
    parser.add_argument('--database', type=str)
    parser.add_argument('--representation', type=str)
    parser.add_argument('--seed', type=seed_type, default=None, nargs="?", help="Random seed")
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

    return parser


def sample_molecules_RNN(database, representation, seed, rnn_type, embedding_size, hidden_size, n_layers,
                         dropout, batch_size, learning_rate, sample_mols, input_file, vocab_file, model_file,
                         output_file, time_file):
    set_seed(seed)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda: {}'.format(torch.cuda.is_available()))

    # load vocabulary
    if representation == 'SELFIES':
        inputs = read_smiles(input_file)
        dataset = SelfiesDataset(selfies=inputs, vocab_file=vocab_file)
        vocab = dataset.vocabulary
    else:
        vocab = Vocabulary(vocab_file=vocab_file)

    # set up model
    model = RNN(vocab,
                rnn_type=rnn_type,
                n_layers=n_layers,
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                dropout=dropout)
    print(vocab.dictionary)

    # load the trained model parameters
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file))
    else:
        model.load_state_dict(torch.load(model_file,
                                         map_location=torch.device('cpu')))

    model.eval()  ## enable evaluation mode

    # wipe the file before writing
    open(output_file, 'w').close()

    # sample a set of SMILES from the final, trained model
    batch_size = 32
    sampled_count = 0
    with tqdm(total=sample_mols) as pbar:
        while sampled_count < sample_mols:
            sampled_smiles, losses = model.sample(batch_size, return_losses=True)
            # increment counter
            sampled_count += batch_size
            # write sampled SMILES
            with open(output_file, 'a+') as f:
                for loss, sm in zip(losses, sampled_smiles):
                    f.write(str(round(loss, 4)) + ',' + sm + '\n')
            pbar.update(batch_size)


def main(args):
    sample_molecules_RNN(database=args.database,
                         representation=args.representation,
                         seed=args.seed,
                         rnn_type=args.rnn_type,
                         embedding_size=args.embedding_size,
                         hidden_size=args.hidden_size,
                         n_layers=args.n_layers,
                         dropout=args.dropout,
                         batch_size=args.batch_size,
                         learning_rate=args.learning_rate,
                         sample_mols=args.sample_mols,
                         input_file=args.input_file,
                         vocab_file=args.vocab_file,
                         model_file=args.model_file,
                         output_file=args.output_file,
                         time_file=args.time_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
