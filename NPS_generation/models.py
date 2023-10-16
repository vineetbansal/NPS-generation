import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, CrossEntropyLoss
from datasets import Variable

class RNN(nn.Module):

    def __init__(self,
                 vocabulary,
                 rnn_type='GRU',
                 embedding_size=128,
                 hidden_size=512,
                 n_layers=3,
                 dropout=0,
                 bidirectional=False,
                 tie_weights=False,
                 nonlinearity=None):
        super(RNN, self).__init__()
        # vocabulary
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        # embedding layer
        padding_token = self.vocabulary.dictionary['<PAD>']
        self.embedding = nn.Embedding(self.vocabulary_size,
                                      embedding_size,
                                      padding_idx=padding_token)
        # RNN itself
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    nonlinearity=nonlinearity,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    batch_first=True)
        else:
            raise ValueError("invalid RNN model type: " + str(rnn_type))
        # dropout
        self.dropout = nn.Dropout(dropout)
        # record parameters
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        # decoder
        linear_dim1 = hidden_size * 2 if bidirectional else hidden_size
        self.decoder = nn.Linear(linear_dim1, self.vocabulary_size)
        # weight tying
        if tie_weights:
            if hidden_size != embedding_size:
                raise ValueError("when using tied weights, hidden_size " + \
                                 "must be equal to embedding_size")
            self.decoder.weight = self.embedding.weight
        self.tie_weights = tie_weights
        # loss function (ignoring padding)
        padding_idx = self.vocabulary.dictionary['<PAD>']
        self.loss_fn = CrossEntropyLoss(ignore_index=padding_idx,
                                        reduction='none')
        # initialize weights
        self.init_weights()
        # move to GPU
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input, hidden):
        """

        input:          (batch_size, seq_len)
        -> embedding layer
        embedded:       (batch_size, seq_len, emb_size)
        -> RNN layer(s)
        output:         (batch_size, seq_len, hidden_size)
        -> reshape for decoder
        reshaped:       (batch_size * seq_len, hidden_size)
        -> output layer
        decoded:        (batch_size * seq_len, vocab_size)
        -> reshape for output
        output:         (batch_size, seq_len, vocab_size)

        Returns: tensor (batch_size, seq_len, vocab_size)
        """
        # word embeddings, with dropout
        embedded = self.embedding(input)
        if self.dropout.p > 0:
            embedded = self.dropout(embedded)
        # mask padding tokens from LSTM
        # packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        # LSTM
        output, hidden = self.rnn(embedded, hidden)
        # undo packing operation before passing to linear output layer
        # unpacked, lengths = pad_packed_sequence(output, batch_first=True)
        # dropout on LSTM output
        if self.dropout.p > 0:
            output = self.dropout(output)
        # reshape for decoder
        reshaped = output.contiguous()
        reshaped = reshaped.view(-1, output.size(2))
        # linear output
        decoded = self.decoder(reshaped)
        # softmax over tokens
        # output = F.log_softmax(decoded, dim=1)
        # reshape and return
        batch_size = input.size(0)
        seq_len = input.size(1)
        vocab_size = len(self.vocabulary)
        return decoded.view(batch_size, seq_len, vocab_size), hidden

    def loss(self, batch, lengths):
        """
        Calculate the cross-entropy loss for a batch of sequences.

        Args:
            batch: (tensor, batch_size * seq_len): a batch of encoded and
              padded sequences
        """
        batch_size = batch.size(0)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        hidden = self.init_hidden(batch_size)

        # calculate log-probabilities over sequence, manually teacher forcing
        log_probs = next(self.parameters()).data.new(batch_size).zero_()
        for step in range(targets.size(1)):
            logits, hidden = self(inputs[:, step].unsqueeze(1), hidden)
            log_probs += self.loss_fn(logits.squeeze(1), targets[:, step])

         # manually average over non-zero lengths
        log_probs = log_probs / lengths.type_as(log_probs)
        return log_probs

    def sample(self, n_seq, max_len=250, return_smiles=True, return_nll=False):
        """
        Sample a batch of sequences from the trained model.

        Args:
            n_seq: number of sequences to sample
            max_len: maximum length of any individual sequence
            return_smiles: (boolean) If true, encode the sequences into
              SMILES
        """
        # get start/stop tokens
        start_token = self.vocabulary.dictionary['SOS']
        stop_token = self.vocabulary.dictionary['EOS']
        # create start token tensor (automatically detect cuda)
        x = next(self.parameters()).data.new(n_seq).\
            fill_(start_token).long()
        # create hidden
        hidden = self.init_hidden(n_seq)
        # create length filler
        # lengths = next(self.parameters()).data.new(n_seq).\
        #     fill_(1).long()

        # sample sequences
        finished = Variable(torch.zeros(n_seq).byte())
        log_probs = Variable(torch.zeros(n_seq))
        sequences = []
        for step in range(max_len):
            logits, hidden = self(x.view(n_seq, 1), hidden)
            prob = F.softmax(logits, dim=2)
            x = torch.multinomial(prob.squeeze(1), 1).view(-1)
            sequences.append(x.view(-1, 1))

            # optionally, calculate NLL too
            log_prob = F.log_softmax(logits, dim=2)
            log_probs += NLLLoss(log_prob.squeeze(1), x)

            # track whether sampling is done for all molecules
            finished = torch.ge(finished + (x == stop_token), 1)
            if torch.prod(finished) == 1:
                break

        # concatenate sequences and decode
        seqs = torch.cat(sequences, 1)
        if return_smiles:
            smiles = [self.vocabulary.decode(seq.cpu().numpy()) for \
                      seq in seqs]
            if return_nll:
                return smiles, log_probs
            else:
                return smiles
        else:
            return sequences

    def init_hidden(self, batch_size):
        """
        Reset the hidden state for a new batch of sequences.

        Note that LSTM takes a tuple (hidden state and cell state; see
        https://pytorch.org/docs/stable/nn.html for the initial hidden state
        tensors that each model expects).

        Returns a tensor of shape [n_layers, batch size, hidden size],
        where n_layers = 2 * n_layers if the model is bidirectional.

        Grab the parameters of the model to instantiate on the right device
        (CPU/GPU) automatically:
        https://discuss.pytorch.org/t/clarifying-init-hidden-method-in-word-\
        language-model-example/13828
        """
        weight = next(self.parameters()).data
        layer_dim = 2 * self.n_layers if self.bidirectional else self.n_layers
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(layer_dim, batch_size,
                                        self.hidden_size).zero_()),
                    Variable(weight.new(layer_dim, batch_size,
                                        self.hidden_size).zero_()))
        else:
            return Variable(weight.new(layer_dim, batch_size,
                                       self.hidden_size).zero_())

    def init_weights(self):
        """
        Xavier initialization of embedding and dense output layer weights.

        Notes:
        - Initialize language model weights?
        """
        init.xavier_normal_(self.embedding.weight)
        if not self.tie_weights:
            init.xavier_normal_(self.decoder.weight)

class OneHotRNN(nn.Module):

    def __init__(self,
                 vocabulary,
                 rnn_type='GRU',
                 hidden_size=512,
                 n_layers=3,
                 dropout=0,
                 bidirectional=False,
                 nonlinearity=None):
        super(OneHotRNN, self).__init__()
        # vocabulary
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        # RNN itself
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=self.vocabulary_size,
                    hidden_size=hidden_size,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(
                    input_size=self.vocabulary_size,
                    hidden_size=hidden_size,
                    nonlinearity=nonlinearity,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    batch_first=True)
        else:
            raise ValueError("invalid RNN model type: " + str(rnn_type))
        # dropout
        self.dropout = nn.Dropout(dropout)
        # record parameters
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        # decoder
        linear_dim1 = hidden_size * 2 if bidirectional else hidden_size
        self.decoder = nn.Linear(linear_dim1, self.vocabulary_size)
        # loss function (ignoring padding)
        padding_idx = self.vocabulary.dictionary['<PAD>']
        self.loss_fn = CrossEntropyLoss(ignore_index=padding_idx,
                                        reduction='none')
        # initialize weights
        self.init_weights()
        # move to GPU
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input, hidden):
        """

        input:          (batch_size, seq_len)
        -> one-hot transformation
        one-hot:        (batch_size, seq_len, vocab_size)
        -> RNN layer(s)
        output:         (batch_size, seq_len, hidden_size)
        -> reshape for decoder
        reshaped:       (batch_size * seq_len, hidden_size)
        -> output layer
        decoded:        (batch_size * seq_len, vocab_size)
        -> reshape for output
        output:         (batch_size, seq_len, vocab_size)

        Returns: tensor (batch_size, seq_len, vocab_size)
        """
        # one-hot embed
        inputs = F.one_hot(input, self.vocabulary_size).float()
        # LSTM
        output, hidden = self.rnn(inputs, hidden)
        # dropout on LSTM output
        if self.dropout.p > 0:
            output = self.dropout(output)
        # reshape for decoder
        reshaped = output.contiguous()
        reshaped = reshaped.view(-1, output.size(2))
        # linear output
        decoded = self.decoder(reshaped)
        # reshape and return
        batch_size = input.size(0)
        seq_len = input.size(1)
        vocab_size = len(self.vocabulary)
        return decoded.view(batch_size, seq_len, vocab_size), hidden

    def loss(self, batch, lengths):
        """
        Calculate the cross-entropy loss for a batch of sequences.

        Args:
            batch: (tensor, batch_size * seq_len): a batch of encoded and
              padded sequences
        """
        batch_size = batch.size(0)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        hidden = self.init_hidden(batch_size)

        # calculate log-probabilities over sequence, manually teacher forcing
        log_probs = next(self.parameters()).data.new(batch_size).zero_()
        for step in range(targets.size(1)):
            logits, hidden = self(inputs[:, step].unsqueeze(1), hidden)
            log_probs += self.loss_fn(logits.squeeze(1), targets[:, step])

         # manually average over non-zero lengths
        log_probs = log_probs / lengths.type_as(log_probs)
        return log_probs

    def sample(self, n_seq, max_len=250, return_smiles=True):
        """
        Sample a batch of sequences from the trained model.

        Args:
            n_seq: number of sequences to sample
            max_len: maximum length of any individual sequence
            return_smiles: (boolean) If true, encode the sequences into
              SMILES
        """
        # get start/stop tokens
        start_token = self.vocabulary.dictionary['SOS']
        stop_token = self.vocabulary.dictionary['EOS']
        # create start token tensor (automatically detect cuda)
        x = next(self.parameters()).data.new(n_seq).\
            fill_(start_token).long()
        # create hidden
        hidden = self.init_hidden(n_seq)

        # sample sequences
        finished = Variable(torch.zeros(n_seq).byte())
        sequences = []
        for step in range(max_len):
            logits, hidden = self(x.view(n_seq, 1), hidden)
            prob = F.softmax(logits, dim=2)
            x = torch.multinomial(prob.squeeze(1), 1).view(-1)
            sequences.append(x.view(-1, 1))

            # track whether sampling is done for all molecules
            finished = torch.ge(finished + (x == stop_token), 1)
            if torch.prod(finished) == 1:
                break

        # concatenate sequences and decode
        seqs = torch.cat(sequences, 1)
        if return_smiles:
            smiles = [self.vocabulary.decode(seq.cpu().numpy()) for \
                      seq in seqs]
            return smiles
        else:
            return sequences

    def init_hidden(self, batch_size):
        """
        Reset the hidden state for a new batch of sequences.

        Note that LSTM takes a tuple (hidden state and cell state; see
        https://pytorch.org/docs/stable/nn.html for the initial hidden state
        tensors that each model expects).

        Returns a tensor of shape [n_layers, batch size, hidden size],
        where n_layers = 2 * n_layers if the model is bidirectional.

        Grab the parameters of the model to instantiate on the right device
        (CPU/GPU) automatically:
        https://discuss.pytorch.org/t/clarifying-init-hidden-method-in-word-\
        language-model-example/13828
        """
        weight = next(self.parameters()).data
        layer_dim = 2 * self.n_layers if self.bidirectional else self.n_layers
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(layer_dim, batch_size,
                                        self.hidden_size).zero_()),
                    Variable(weight.new(layer_dim, batch_size,
                                        self.hidden_size).zero_()))
        else:
            return Variable(weight.new(layer_dim, batch_size,
                                       self.hidden_size).zero_())

    def init_weights(self):
        """
        Xavier initialization of embedding and dense output layer weights.

        Notes:
        - Initialize language model weights?
        """
        init.xavier_normal_(self.decoder.weight)

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Obtained from REINVENT source code.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss

class EarlyStopping():
    """
    Monitor the training process to stop training early if the model shows
    evidence of beginning to overfit the validation dataset, and save the
    best model.

    Note that patience here is measured in steps, rather than in epochs,
    because the size of an epoch will not be consistent if the size of the
    dataset changes.

    Inspired by:
    https://github.com/Bjarten/early-stopping-pytorch
    https://github.com/fastai/fastai/blob/master/courses/dl2/imdb_scripts/finetune_lm.py
    """

    def __init__(self, patience=100):
        """
        Args:
            model: the PyTorch model being trained
            output_file: (str) location to save the trained model
            patience: (int) if the validation loss fails to improve for this
              number of consecutive batches, training will be stopped
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.step_at_best = 0
        self.stop = False
        print("instantiated early stopping with patience=" + \
              str(self.patience))

    def __call__(self, val_loss, model, output_file, step_idx):
        # do nothing if early stopping is disabled
        if self.patience > 0:
            if self.best_loss is None:
                self.best_loss = val_loss
                self.step_at_best = step_idx
                self.save_model(model, output_file)
            elif val_loss >= self.best_loss:
                # loss is not decreasing
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop = True
                    print("stopping early with best loss " + \
                          str(self.best_loss))
            else:
                # loss is decreasing
                self.best_loss = val_loss
                self.step_at_best = step_idx
                ## reset counter
                self.counter = 0
                self.save_model(model, output_file)

    def save_model(self, model, output_file):
        torch.save(model.state_dict(), output_file)
