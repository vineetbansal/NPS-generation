import logging
import os
import os.path
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rdkit import rdBase

from clm.datasets import (
    SmilesDataset,
    SelfiesDataset,
    SmilesCollate,
    SmilesDescriptorsCollate,
)
from clm.models import RNN, ConditionalRNN
from clm.loggers import EarlyStopping, track_loss, print_update
from clm.functions import read_file, write_smiles

# suppress Chem.MolFromSmiles error output
rdBase.DisableLog("rdApp.error")
logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--representation",
        type=str,
        default="SMILES",
        help="Molecular representation format (one of: SMILES/SELFIES)",
    )

    parser.add_argument(
        "--rnn_type", type=str, help="Type of RNN used (e.g., LSTM, GRU)"
    )

    parser.add_argument(
        "--embedding_size", type=int, help="Size of the embedding layer"
    )

    parser.add_argument("--hidden_size", type=int, help="Size of the hidden layers")

    parser.add_argument("--n_layers", type=int, help="Number of layers in the RNN")

    parser.add_argument("--dropout", type=float, help="Dropout rate for the RNN")

    parser.add_argument("--batch_size", type=int, help="Batch size for training")

    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--max_epochs", type=int, help="Maximum number of epochs for training"
    )

    parser.add_argument("--patience", type=int, help="Patience for early stopping")

    parser.add_argument(
        "--log_every_steps", type=int, help="Logging frequency in steps"
    )

    parser.add_argument(
        "--log_every_epochs", type=int, help="Logging frequency in epochs"
    )

    parser.add_argument(
        "--sample_mols", type=int, help="Number of molecules to sample for evaluation"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file path for training data",
    )

    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="Output path for the vocabulary file ({fold} is populated automatically)",
    )

    parser.add_argument(
        "--smiles_file",
        type=str,
        default=None,
        help="File path for additional SMILES data (optional)",
    )

    parser.add_argument(
        "--model_file", type=str, help="File path to save the trained model"
    )

    parser.add_argument(
        "--loss_file", type=str, help="File path to save the training loss data"
    )

    parser.add_argument(
        "--conditional_rnn", action="store_true", help="Activate Conditional RNN model"
    )
    parser.add_argument(
        "--minmax_descriptor_file",
        type=str,
        default=None,
        help="File path for storing min and max of all the descriptors which would be responsible as inputs for sampling",
    )

    return parser


def load_dataset(representation, input_file, vocab_file, conditional_rnn):
    dataset_class, collate_class = {
        ("SMILES", False): (SmilesDataset, SmilesCollate),
        ("SMILES", True): (SmilesDataset, SmilesDescriptorsCollate),
        ("SELFIES", False): (SelfiesDataset, SmilesCollate),
    }[(representation, conditional_rnn)]

    inputs = read_file(input_file, smile_only=True)
    return dataset_class(inputs, vocab_file=vocab_file, collate_class=collate_class)


def training_step(batch, model, optim, dataset, batch_size):
    loss = model.loss(batch)
    optim.zero_grad()
    loss.backward()
    optim.step()
    validation = dataset.get_validation(batch_size)
    validation_loss = model.loss(validation)
    return validation_loss


def sample_and_write_smiles(model, sample_mols, batch_size, smiles_file):
    sampled_smiles = []
    with tqdm(total=sample_mols) as pbar:
        while len(sampled_smiles) < sample_mols:
            new_smiles = model.sample(batch_size, return_smiles=True)
            sampled_smiles.extend(new_smiles)
            pbar.update(len(new_smiles))
    write_smiles(sampled_smiles, smiles_file)


def train_models_RNN(
    representation,
    rnn_type,
    embedding_size,
    hidden_size,
    n_layers,
    dropout,
    batch_size,
    learning_rate,
    max_epochs,
    patience,
    log_every_steps,
    log_every_epochs,
    sample_mols,
    input_file,
    vocab_file,
    smiles_file,
    model_file,
    loss_file,
    conditional_rnn=False,
    minmax_descriptor_file=None,
):

    os.makedirs(os.path.dirname(os.path.abspath(model_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(loss_file)), exist_ok=True)
    summary_writer = SummaryWriter(comment="_conditional" if conditional_rnn else "")

    dataset = load_dataset(representation, input_file, vocab_file, conditional_rnn)

    logger.info(dataset.vocabulary.dictionary)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate
    )

    if conditional_rnn:
        model = ConditionalRNN(
            dataset.vocabulary,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_descriptors=dataset.collate.n_descriptors,
        )
    else:
        model = RNN(
            dataset.vocabulary,
            rnn_type=rnn_type,
            n_layers=n_layers,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

    optim = Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08, lr=learning_rate)
    if conditional_rnn:
        early_stop = EarlyStopping(
            patience=patience, n_descriptors=dataset.collate.n_descriptors
        )
    else:
        early_stop = EarlyStopping(patience=patience, n_descriptors=None)

    for epoch in range(max_epochs):
        for batch_no, batch in tqdm(enumerate(loader), total=len(loader)):
            loss = training_step(batch, model, optim, dataset, batch_size)
            validation_loss = loss.detach()
            if conditional_rnn:
                _, _, descriptors = batch

                batch_min = descriptors.min(dim=0).values
                batch_max = descriptors.max(dim=0).values
            else:
                batch_min = None
                batch_max = None

            loop_count = (epoch * len(loader)) + batch_no + 1
            if loop_count % log_every_steps == 0 or (
                log_every_epochs and (batch_no + 1) == len(loader)
            ):
                track_loss(
                    loss_file,
                    epoch + 1,
                    loop_count,
                    value=[loss.item(), validation_loss.item()],
                    writer=summary_writer,
                )
                if conditional_rnn:
                    print_update(
                        model,
                        epoch,
                        batch_no + 1,
                        loss.item(),
                        validation_loss.item(),
                        batch_size,
                        batch[2],
                    )
                else:
                    print_update(
                        model, epoch, batch_no + 1, loss.item(), validation_loss.item()
                    )
            early_stop(
                validation_loss.item(),
                model,
                model_file,
                loop_count,
                batch_min,
                batch_max,
            )

            if early_stop.stop:
                logging.info("Early stopping triggered.")
                break

        if early_stop.stop:
            break

    if log_every_epochs or log_every_steps:
        track_loss(
            loss_file,
            epoch=[None],
            batch_no=early_stop.step_at_best,
            value=[early_stop.best_loss],
            outcome=["training loss"],
            writer=summary_writer,
        )

    model.load_state_dict(torch.load(model_file))
    if conditional_rnn and minmax_descriptor_file:
        early_stop.generate_csv(minmax_descriptor_file)
    model.eval()
    if smiles_file:
        sample_and_write_smiles(model, sample_mols, batch_size, smiles_file)


def main(args):
    train_models_RNN(
        representation=args.representation,
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
        conditional_rnn=args.conditional_rnn,
        minmax_descriptor_file=args.minmax_descriptor_file,
    )
