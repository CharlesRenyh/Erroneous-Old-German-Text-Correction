import os
import torch
import torchtext.vocab as Vocab
import collections
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, data_root, max_length):
        self.data_root = data_root
        self.max_length = max_length
        self.samples = []
        self._init_dataset()
        self.vocab = self.get_vocab()
        self.features = self.preprocess_text(self.vocab)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        word = self.samples[idx]
        return self.preprocess_text(word)

    def _init_dataset(self):
        with open('AfterTransfer.txt', 'rb') as b:
            for word in b.read().decode("utf-8", "ignore").split():

                if len(word) > self.max_length:
                    word = word[:self.max_length-1] + '\0'
                else:
                    word = word + '\0' * (self.max_length - len(word))

                self.samples.append(word)

    def get_vocab(self):
        processed_data = self.samples
        counter = collections.Counter([tk for st in processed_data for tk in st])
        return Vocab.Vocab(counter, min_freq=1)

    def preprocess_text(self, vocab):
        processed_data = self.samples
        self.vocab = vocab
        print(vocab)
        features_sample = torch.tensor([[vocab.stoi[word] for word in words] for words in processed_data])
        return features_sample


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_root = 'Data/'
    max_length = 6
    dataset = TextDataset(data_root, max_length)
    print(dataset[100])

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(next(iter(dataloader)))

