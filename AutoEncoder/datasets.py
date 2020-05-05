import os
import torch
import numpy as np
import collections
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TextDataset(Dataset):

    def __init__(self, data_root, max_length):
        self.data_root = data_root
        self.max_length = max_length
        self.samples = []
        self._init_dataset()
        self.charset = list(self.get_charset())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence = self.samples[idx]
        ind_list = []
        for char in sentence:
            ind_list.append(self.charset.index(char))
        array_item = np.asarray(ind_list)
        item = torch.LongTensor(array_item)
        return item

    def _init_dataset(self):
        with open('../AfterTransfer.txt', 'rb') as b:
            for sentence in b.read().decode("utf-8", "ignore").splitlines():

                if len(sentence) > self.max_length:
                    sentence = sentence[:self.max_length-1] + '\0'
                else:
                    sentence = sentence + '\0' * (self.max_length - len(sentence))

                self.samples.append(sentence)

    def get_charset(self):
        charset = set()
        processed_data = self.samples
        for word in processed_data:
            for i in range(self.max_length):
                charset.add(word[i])
        return charset


'''
    def to_one_hot(self, word):
        data_to_use = self.charset
        n = len(data_to_use)
        index = []
        for char in word:
            index.append(data_to_use.index(char))
        convert_torch = torch.LongTensor(index)
        one_hot = F.one_hot(convert_torch, num_classes=n)
        return one_hot
'''


if __name__ == '__main__':
    data_root = 'Data/'
    max_length = 50
    dataset = TextDataset(data_root, max_length)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(next(iter(dataloader)))
    print(len(dataset.charset))

'''
input(seq_len, batch, input_size) 
h0(num_layers * num_directions, batch, hidden_size) 
c0(num_layers * num_directions, batch, hidden_size)
'''
