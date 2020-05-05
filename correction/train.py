import math
import time


# def train(model: nn.Module,
#           iterator: BucketIterator,
#           optimizer: optim.Optimizer,
#           criterion: nn.Module,
#           clip: float):
#
#     model.train()
#
#     epoch_loss = 0
#
#     for _, batch in enumerate(iterator):
#
#         src = batch.src
#         trg = batch.trg
#
#         optimizer.zero_grad()
#
#         output = model(src, trg)
#
#         output = output[1:].view(-1, output.shape[-1])
#         trg = trg[1:].view(-1)
#
#         loss = criterion(output, trg)
#
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#
#         optimizer.step()
#
#         epoch_loss += loss.item()
#
#     return epoch_loss / len(iterator)
#
#
# def evaluate(model: nn.Module,
#              iterator: BucketIterator,
#              criterion: nn.Module):
#
#     model.eval()
#
#     epoch_loss = 0
#
#     with torch.no_grad():
#
#         for _, batch in enumerate(iterator):
#
#             src = batch.src
#             trg = batch.trg
#
#             output = model(src, trg, 0) #turn off teacher forcing
#
#             output = output[1:].view(-1, output.shape[-1])
#             trg = trg[1:].view(-1)
#
#             loss = criterion(output, trg)
#
#             epoch_loss += loss.item()
#
#     return epoch_loss / len(iterator)
#
#
# def epoch_time(start_time: int,
#                end_time: int):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs
#
#
# N_EPOCHS = 10
# CLIP = 1
#
# best_valid_loss = float('inf')
#
# for epoch in range(N_EPOCHS):
#
#     start_time = time.time()
#
#     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#     valid_loss = evaluate(model, valid_iterator, criterion)
#
#     end_time = time.time()
#
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#
#     print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
#
# test_loss = evaluate(model, test_iterator, criterion)
#
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from correction_model import Encoder, Attention,Decoder,Seq2Seq
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.optim as optim

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
word2id["PAD"] = 0
id2word = {i:w for i, w in enumerate(total_word)}
id2word[0] = "PAD"
max_len = 15
data = []
for sen1, sen2 in zip(wrong, right):
    org = [word2id[w] for w in sen1.split()]
    trt = [word2id[w] for w in sen2.split()]
    org += (max_len - len(org)) * [len(word2id)]
    trt += (max_len - len(trt)) * [len(word2id)]
    data.append([org, trt])




class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return [torch.LongTensor(self.data[idx][0]).view(1, -1), torch.LongTensor(self.data[idx][1]).view(1, -1)]

    def __len__(self):
        return len(self.data)
dataset = TextDataset(data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = Encoder(len(word2id)+1, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(len(word2id)+1, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

# print(model)
def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters(),lr=0.0001)

EPOCHS = 20
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(20):
    for step, (org, trt) in enumerate(dataloader):
        output = model(org.view(-1, 16), trt.view(-1, 16))
        loss = loss_fn(output.view(-1, output.size(-1)), trt.view(-1))
        if step % 10 == 0:
            print("train loss:", loss.item())
