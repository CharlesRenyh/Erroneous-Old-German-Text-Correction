from collections import deque
import string
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import TextDataset
from model import LSTMAE

data_root = '/Users/charlesren/Downloads/AI_lab_data/Data'
max_length = 6
batch_size = 50
num_epochs = 100
learning_rate = 0.0003
print_interval = 100

dataset = TextDataset(data_root, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

losses = deque([], maxlen=print_interval)

input_size = 62
hidden_size = 16
model = LSTMAE(input_size, hidden_size, num_layers=1, isCuda=False)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

losses = deque([], maxlen=print_interval)

for epoch in range(num_epochs):
    for batch_i, samples in enumerate(dataloader):
        model.zero_grad()
        word_set = samples
        #print(word_set.size())

        loss = 0.

        for word_ind in range(batch_size):
            word = word_set[word_ind]
            #print(t_char)

            output = model(word)
            target = word.clone()
            #print(target)
            loss += loss_func(output, target)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
