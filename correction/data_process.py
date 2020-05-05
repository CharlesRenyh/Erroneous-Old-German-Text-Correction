# from torchtext.data import Field, BucketIterator
# import torch
#
# SRC = Field(tokenize = "spacy",
#             tokenizer_language="de",
#             init_token = '<sos>',
#             eos_token = '<eos>',
#             lower = True)
#
# TRG = Field(tokenize = "spacy",
#             tokenizer_language="en",
#             init_token = '<sos>',
#             eos_token = '<eos>',
#             lower = True)
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# BATCH_SIZE = 128
#
# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data),
#     batch_size = BATCH_SIZE,
#     device = device)
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
ori_path = '../Data/'
for book in os.listdir(ori_path):
    with open(ori_path + book, 'r', encoding='latin-1') as t:
        lines = t.readlines()
        n = len(lines)
        wrong = []
        right = []
        for i in range(1, n, 3):
            wrong.append(lines[i])
            right.append(lines[i-1])
total_word = []
for sen1, sen2 in zip(wrong, right):
    total_word.extend(sen1.split())
    total_word.extend(sen2.split())
total_word = list(set(total_word))
word2id = {w:i for i, w in enumerate(total_word)}
id2word = {i:w for i, w in enumerate(total_word)}
data = []
for sen1, sen2 in zip(wrong, right):
    data.append([[word2id[w] for w in sen1.split()], [word2id[w] for w in sen2.split()]])




class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return [torch.LongTensor(self.data[idx][0]).view(1, -1), torch.LongTensor(self.data[idx][1]).view(1, -1)]

    def __len__(self):
        return len(self.data)
dataset = TextDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

