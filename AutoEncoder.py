import collections
import os
import random
import numpy as np
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
from tqdm import tqdm
import sys


def read_text(data_name):
    data = []
    with open(data_name, 'rb') as f:
        review = f.read().decode('latin-1').replace('\n', '')
        data.append(review)
    random.shuffle(data)
    return data


train_data = read_text('AfterTransfer.txt')
test_data = read_text('ForTest.txt')


def get_tokenized_text(data):
    """
    data: list of [string]
    """
    def tokenizer(text):
        return [tok for tok in text.split(' ')]
    return [tokenizer(review) for review in data]


def get_vocab_text(data):
    tokenized_data = get_tokenized_text(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=10)  # when freq = 1, vocab = 81349


vocab = get_vocab_text(train_data)
vocab_size = len(vocab)
print('# words in vocab:', vocab_size)


def preprocess_text(data, vocab):
    tokenized_data = get_tokenized_text(data)
    features = torch.tensor([[vocab.stoi[word] for word in words] for words in tokenized_data])
    return features


batch_size = 64
train_set = Data.TensorDataset(preprocess_text(train_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
print(type(train_set))
print(type(train_iter))
print(type(preprocess_text(train_data, vocab)))

embedding_dim = 500
torch.manual_seed(1)
word_embedding = nn.Embedding(vocab_size, embedding_dim)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = word_embedding
        self.isCuda = isCuda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, input):
        tt = torch.cuda if self.isCuda else torch
        h0 = tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size)
        c0 = tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size)
        encoded_input, hidden = self.lstm(input, (h0, c0))

        encoded_input = self.relu(encoded_input)
        return encoded_input


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = word_embedding
        self.isCuda = isCuda
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        h0 = tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size)
        c0 = tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size)
        decoded_output, hidden = self.lstm(encoded_input, (h0, c0))
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.embedding = word_embedding
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, isCuda)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output


num_layers = 2
AutoEncoder = LSTMAE(vocab_size, embedding_dim, num_layers, False)
num_epochs = 10
LR = 0.01


optimizer = torch.optim.Adam(AutoEncoder.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    i = 0
    for batch in train_iter:
        i += 1
        BatchToUse = batch[0].unsqueeze(0)
        print(BatchToUse.size())
        original = BatchToUse
        decoded = AutoEncoder(BatchToUse)
        loss = loss_func(decoded, original)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
