"""
Datasets used by PyTorch for language modelling of chemical structures.
"""

import numpy as np
import re
import selfies as sf
import torch
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from torch.utils.data import Dataset
from NPS_generation.functions import read_smiles

class SmilesDataset(Dataset):
    """
    A dataset of chemical structures, provided in SMILES format.
    """

    def __init__(self, smiles=None, smiles_file=None, vocab_file=None,
                 training_split=0.9):
        """
        Can be initiated from either a list of SMILES, or a line-delimited
        file.

        Args:
            smiles (list): the complete set of SMILES that constitute the
              training dataset
            smiles_file (string): line-delimited file containing the complete
              set of SMILES that constitute the training dataset
            vocab_file (string): line-delimited file containing all tokens to
              be used in the vocabulary
            training_split (numeric): proportion of the dataset to withhold for
              validation loss calculation
        """
        if smiles:
            self.smiles = smiles
        elif smiles_file:
            self.smiles = read_smiles(smiles_file)
        else:
            raise ValueError("must provide SMILES list or file to" + \
                             " instantiate SmilesDataset")

        # create vocabulary
        if vocab_file:
            self.vocabulary = Vocabulary(vocab_file=vocab_file)
        else:
            self.vocabulary = Vocabulary(smiles=self.smiles)

        # split into training and validation sets
        np.random.seed(0)
        n_smiles = len(self.smiles)
        split = np.random.choice(range(n_smiles),
                                 size=int(n_smiles * training_split),
                                 replace=False)
        self.training = [self.smiles[idx] for idx in \
                         range(len(self.smiles)) if idx in split]
        self.validation = [self.smiles[idx] for idx in \
                           range(len(self.smiles)) if not idx in split]

    def get_validation(self, n_smiles):
        validation_size = len(self.validation)
        idxs = np.random.choice(np.asarray(range(validation_size)),
                                size=n_smiles)
        encoded = [Variable(self.vocabulary.encode(self.vocabulary.tokenize(
                   self.validation[idx]))) for idx in idxs]
        collate_fn = SmilesCollate(self.vocabulary)
        return collate_fn(encoded)

    def __len__(self):
        return len(self.training)

    def __getitem__(self, idx):
        return Variable(self.vocabulary.encode(
                self.vocabulary.tokenize(self.training[idx])))

    def __str__(self):
        return "dataset containing " + str(len(self)) + \
            " SMILES with a vocabulary of " + str(len(self.vocabulary)) + \
            " characters"

class SelfiesDataset(Dataset):
    """
    A dataset of chemical structures, provided in SELFIES format.
    """

    def __init__(self, selfies=None, selfies_file=None, vocab_file=None,
                 training_split=0.9):
        """
        Can be initiated from either a list of SELFIES, or a line-delimited
        file.

        Args:
            selfies (list): the complete set of SELFIES that constitute the
              training dataset
            selfies_file (string): line-delimited file containing the complete
              set of SELFIES that constitute the training dataset
            training_split (numeric): proportion of the dataset to withhold for
              validation loss calculation
        """
        if selfies is not None:
            self.selfies = selfies
        elif selfies_file is not None:
            self.selfies = read_smiles(selfies_file)
        else:
            raise ValueError("must provide SELFIES list or file to" + \
                             " instantiate SelfiesDataset")

        # create vocabulary
        if vocab_file:
            self.vocabulary = Vocabulary(vocab_file=vocab_file)
        else:
            self.vocabulary = SelfiesVocabulary(selfies=self.selfies)

        # split into training and validation sets
        np.random.seed(0)
        n_selfies = len(self.selfies)
        split = np.random.choice(range(n_selfies),
                                 size=int(n_selfies * training_split),
                                 replace=False)
        self.training = [self.selfies[idx] for idx in \
                         range(len(self.selfies)) if idx in split]
        self.validation = [self.selfies[idx] for idx in \
                           range(len(self.selfies)) if not idx in split]

    def get_validation(self, n_selfies):
        validation_size = len(self.validation)
        idxs = np.random.choice(np.asarray(range(validation_size)),
                                size=n_selfies)
        encoded = [Variable(self.vocabulary.encode(self.vocabulary.tokenize(
                   self.validation[idx]))) for idx in idxs]
        collate_fn = SmilesCollate(self.vocabulary)
        return collate_fn(encoded)

    def __len__(self):
        return len(self.training)

    def __getitem__(self, idx):
        return Variable(self.vocabulary.encode(
                self.vocabulary.tokenize(self.training[idx])))

    def __str__(self):
        return "dataset containing " + str(len(self)) + \
            " SELFIES with a vocabulary of " + str(len(self.vocabulary)) + \
            " characters"

class SmilesCollate():
    """
    Collate a list of SMILES tensors, with variable lengths, into a tensor.

    Code adapted from: https://www.codefull.org/2018/11/use-pytorchs-\
    dataloader-with-variable-length-sequences-for-lstm-gru/

    Args:
        batch (list): a list of numeric tensors, each derived from a single
          SMILES string, where the value at each position in the tensor
          is the index of the SMILES token in the vocabulary dictionary

    Return:
        a tensor of dimension (batch_size, seq_len) containing encoded and
          padded sequences
    """
    def __init__(self, vocabulary):
        padding_token = vocabulary.dictionary['<PAD>']
        self.padding_token = padding_token

    def __call__(self, batch):
        # sort batch in descending order of length
        sorted_batch = sorted(batch, key=lambda x: x.shape[0], reverse=True)
        # get each sequence and pad it
        sequences = [x for x in sorted_batch]
        padded = Variable(rnn_utils.pad_sequence(
                sequences, padding_value=self.padding_token).\
                transpose_(1, 0))
        # also store lengths of each sequence (needed to unpad)
        lengths = torch.LongTensor([len(x) for x in sequences])
        return padded, lengths

class Vocabulary(object):
    """
    Handles encoding and decoding of SMILES to and from one-hot vectors.
    """

    def __init__(self, smiles=None, smiles_file=None, vocab_file=None):
        """
        Can be initiated from either a list of SMILES, or a line-delimited
        SMILES file, or a file containing only tokens.

        Args:
            smiles (list): the complete set of SMILES that constitute the
              training dataset
            smiles_file (string): line-delimited file containing the complete
              set of SMILES that constitute the training dataset
            vocab_file (string): line-delimited file containing all tokens to
              be used in the vocabulary
        """
        if vocab_file is not None:
            # read tokens from file, and add to vocabulary
            self.characters = read_smiles(vocab_file)
        else:
            # read SMILES
            if smiles is not None:
                self.smiles = smiles
            elif smiles_file is not None:
                self.smiles = read_smiles(smiles_file)
            else:
                raise ValueError("must provide SMILES list or file to" + \
                                 " instantiate Vocabulary")
            # tokenize all SMILES in the input and add all tokens to vocabulary
            all_chars = [self.tokenize(sm) for sm in self.smiles]
            self.characters = list(set(chain(*all_chars)))

        # add padding token
        if not '<PAD>' in self.characters:
            # ... unless reading a padded vocabulary from file
            self.characters.append('<PAD>')
        
        # create dictionaries
        self.dictionary = {key: idx for idx, key in enumerate(self.characters)}
        self.reverse_dictionary = {value: key for key, value in \
                                   self.dictionary.items()}

    """
    Regular expressions used to tokenize SMILES strings; borrowed from
    https://github.com/undeadpixel/reinvent-randomized/blob/master/models/vocabulary.py
    """
    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, smiles):
        """
        Convert a SMILES string into a sequence of tokens.
        """
        def split_by(smiles, regexps):
            if not regexps:
                return list(smiles)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(smiles)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(smiles, self.REGEXP_ORDER)
        tokens = ["SOS"] + tokens + ["EOS"]
        return tokens

    def encode(self, tokens):
        """
        Encode a series of tokens into a (numeric) tensor.
        """
        vec = torch.zeros(len(tokens))
        for idx, token in enumerate(tokens):
            vec[idx] = self.dictionary[token]
        return vec.long()

    def decode(self, sequence):
        """
        Decode a series of tokens back to a SMILES.
        """
        chars = []
        for i in sequence:
            if i == self.dictionary['EOS']:
                break
            if i != self.dictionary['SOS']:
                chars.append(self.reverse_dictionary[i])
        smiles = "".join(chars)
        # smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def write(self, output_file):
        """
        Write the list of tokens in a vocabulary to a line-delimited file.
        """
        with open(output_file, 'w') as f:
            for char in self.characters:
                f.write(char + '\n')

    def __len__(self):
        return len(self.characters)

    def __str__(self):
        return "vocabulary containing " + str(len(self)) + " characters: " + \
            format(self.characters)

class SelfiesVocabulary(object):
    """
    Handles encoding and decoding of SELFIES to and from one-hot vectors.
    """

    def __init__(self, selfies=None, selfies_file=None, vocab_file=None):
        """
        Can be initiated from either a list of SELFIES, or a line-delimited
        SELFIES file.

        Args:
            selfies (list): the complete set of SELFIES that constitute the
              training dataset
            selfies_file (string): line-delimited file containing the complete
              set of SELFIES that constitute the training dataset
            vocab_file (string): line-delimited file containing all tokens to
              be used in the vocabulary
        """
        if vocab_file is not None:
            # read tokens from file, and add to vocabulary
            all_chars = read_smiles(vocab_file)
            # prevent chain popping open multi-character tokens
            self.characters = list(set(chain(*[[char] for char in all_chars])))
        else:
            # read SMILES
            if selfies is not None:
                self.selfies = selfies
            elif selfies_file is not None:
                self.selfies = read_smiles(selfies_file)
            else:
                raise ValueError("must provide SELFIES list or file to" + \
                                 " instantiate Vocabulary")
            # tokenize all SMILES in the input and add all tokens to vocabulary
            alphabet = sorted(list(sf.get_alphabet_from_selfies(self.selfies)))
            self.characters = alphabet

        # add padding token
        self.characters.append('<PAD>')
        # add SOS/EOS tokens
        self.characters.append('SOS')
        self.characters.append('EOS')
        # create dictionaries
        self.dictionary = {key: idx for idx, key in enumerate(self.characters)}
        self.reverse_dictionary = {value: key for key, value in \
                                   self.dictionary.items()}

    def tokenize(self, selfie):
        """
        Convert a SELFIES string into a sequence of tokens.
        """
        tokens = list(sf.split_selfies(selfie))
        tokens = ["SOS"] + tokens + ["EOS"]
        return tokens

    def encode(self, tokens):
        """
        Encode a series of tokens into a (numeric) tensor.
        """
        vec = torch.zeros(len(tokens))
        for idx, token in enumerate(tokens):
            vec[idx] = self.dictionary[token]
        return vec.long()

    def decode(self, sequence):
        """
        Decode a series of tokens back to a SELFIES.
        """
        chars = []
        for i in sequence:
            if i == self.dictionary['EOS']:
                break
            if i != self.dictionary['SOS']:
                chars.append(self.reverse_dictionary[i])
        smiles = "".join(chars)
        return smiles

    def write(self, output_file):
        """
        Write the list of tokens in a vocabulary to a line-delimited file.
        """
        with open(output_file, 'w') as f:
            for char in self.characters:
                f.write(char + '\n')

    def __len__(self):
        return len(self.characters)

    def __str__(self):
        return "vocabulary containing " + str(len(self)) + " characters: " + \
            format(self.characters)

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU.
       Obtained from REINVENT source code."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)
