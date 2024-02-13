import numpy as np
import re
import selfies as sf
import torch
from itertools import chain
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from functions import read_smiles

class Vocabulary(object):
    def __init__(self, smiles=None, vocab_file=None):
        if vocab_file is not None:
            self.characters = read_smiles(vocab_file)
        else:
            self.smiles = smiles
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
        vec = torch.zeros(len(tokens))
        for idx, token in enumerate(tokens):
            vec[idx] = self.dictionary[token]
        return vec.long()
    
    def decode(self, sequence):
        chars = []
        for i in sequence:
            if i == self.dictionary['EOS']:
                break
            if i != self.dictionary['SOS']:
                chars.append(self.reverse_dictionary[i])
        smiles = "".join(chars)
        return smiles
    
    def write(self, output_file):
        with open(output_file, 'w') as f:
            for char in self.characters:
                f.write(char + '\n')
    
    def __len__(self):
        return len(self.characters)
    
    def __str__(self):
        return "vocabulary containing " + str(len(self)) + " characters: " + \
            format(self.characters)

class SelfiesVocabulary(object):
    def __init__(self, selfies=None, vocab_file=None):
        if vocab_file is not None:
            self.characters = read_smiles(vocab_file)
        else:
            self.selfies = selfies
            # tokenize all SMILES in the input and add all tokens to vocabulary
            alphabet = sorted(list(sf.get_alphabet_from_selfies(self.selfies)))
            self.characters = alphabet
            # add SOS, EOS
            self.characters.extend(['SOS', 'EOS'])
        
        # add padding token
        if not '<PAD>' in self.characters:
            # ... unless reading a padded vocabulary from file
            self.characters.append('<PAD>')
        
        # create dictionaries
        self.dictionary = {key: idx for idx, key in enumerate(self.characters)}
        self.reverse_dictionary = {value: key for key, value in \
                                   self.dictionary.items()}
    
    def tokenize(self, selfie):
        tokens = list(sf.split_selfies(selfie))
        tokens = ["SOS"] + tokens + ["EOS"]
        return tokens
    
    def encode(self, tokens):
        vec = torch.zeros(len(tokens))
        for idx, token in enumerate(tokens):
            vec[idx] = self.dictionary[token]
        return vec.long()
    
    def decode(self, sequence):
        chars = []
        for i in sequence:
            if i == self.dictionary['EOS']:
                break
            if i != self.dictionary['SOS']:
                chars.append(self.reverse_dictionary[i])
        smiles = "".join(chars)
        return smiles
    
    def write(self, output_file):
        with open(output_file, 'w') as f:
            for char in self.characters:
                f.write(char + '\n')
    
    def __len__(self):
        return len(self.characters)
    
    def __str__(self):
        return "vocabulary containing " + str(len(self)) + " characters: " + \
            format(self.characters)

class SmilesDataset(Dataset):
    def __init__(self, 
                 smiles,
                 max_len=None,
                 vocab_file=None,
                 training_split=0.9):
        # shuffle the SMILES
        self.smiles = smiles
        np.random.shuffle(self.smiles)
                
        # create vocabulary or else read from file
        if vocab_file:
            self.vocabulary = Vocabulary(vocab_file=vocab_file)
        else:
            self.vocabulary = Vocabulary(smiles=self.smiles)
        
        # remove SMILES greater than max_len
        self.max_len = max_len
        if self.max_len is not None:
            self.smiles = [sm for sm in self.smiles if \
                           len(self.vocabulary.tokenize(sm)) <= self.max_len]
        
        # split out a validation set
        n_smiles = len(self.smiles)
        border = int(n_smiles * training_split)
        self.training_set = self.smiles[:border]
        self.validation_set = self.smiles[border:]
        
        # define collate function
        self.collate = SmilesCollate(self.vocabulary)
    
    def __len__(self):
        return len(self.training_set)
    
    def __getitem__(self, idx):
        smiles = self.training_set[idx]
        tokenized = self.vocabulary.tokenize(smiles)
        encoded = self.vocabulary.encode(tokenized)
        return encoded
    
    def get_validation(self, n_smiles):
        smiles = np.random.choice(self.validation_set, n_smiles)
        tokenized = [self.vocabulary.tokenize(sm) for sm in smiles]
        encoded = [self.vocabulary.encode(tk) for tk in tokenized]
        return self.collate(encoded)
    
    def __str__(self):
        return "dataset containing " + str(len(self)) + \
            " SMILES with a vocabulary of " + str(len(self.vocabulary)) + \
            " characters"

class SmilesCollate():
    def __init__(self, vocabulary):
        self.padding_token = vocabulary.dictionary['<PAD>']
    
    def __call__(self, encoded):
        padded = pad_sequence(encoded, padding_value=self.padding_token)
        lengths = [len(seq) for seq in encoded]
        return padded, lengths

class SelfiesDataset(Dataset):
    def __init__(self, 
                 selfies,
                 max_len=None,
                 vocab_file=None,
                 training_split=0.9):
        # shuffle the SELFIES
        self.selfies = selfies
        np.random.shuffle(self.selfies)
         
        # create vocabulary or else read from file
        if vocab_file:
            self.vocabulary = SelfiesVocabulary(vocab_file=vocab_file)
        else:
            self.vocabulary = SelfiesVocabulary(selfies=self.selfies)
        
        # remove SMILES greater than max_len
        self.max_len = max_len
        if self.max_len is not None:
            self.selfies = [sf for sf in self.selfies if \
                            len(self.vocabulary.tokenize(sf)) <= self.max_len]
        
        # split out a validation set
        n_selfies = len(self.selfies)
        border = int(n_selfies * training_split)
        self.training_set = self.selfies[:border]
        self.validation_set = self.selfies[border:]
        
        # define collate function
        self.collate = SmilesCollate(self.vocabulary)
    
    def __len__(self):
        return len(self.training_set)
    
    def __getitem__(self, idx):
        selfies = self.training_set[idx]
        tokenized = self.vocabulary.tokenize(selfies)
        encoded = self.vocabulary.encode(tokenized)
        return encoded
    
    def get_validation(self, n_selfies):
        selfies = np.random.choice(self.validation_set, n_selfies)
        tokenized = [self.vocabulary.tokenize(sf) for sf in selfies]
        encoded = [self.vocabulary.encode(tk) for tk in tokenized]
        return self.collate(encoded)
    
    def __str__(self):
        return "dataset containing " + str(len(self)) + \
            " SELFIES with a vocabulary of " + str(len(self.vocabulary)) + \
            " characters"
