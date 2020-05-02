from collections import deque
import string
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import TextDataset
from model import LSTMAE
from generate import generate

data_root = '/Users/charlesren/Downloads/AI_lab_data/Data'
max_length = 50
batch_size = 50
num_epochs = 100
learning_rate = 0.0003
print_interval = 100

dataset = TextDataset(data_root, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

losses = deque([], maxlen=print_interval)

input_size = max_length
hidden_size = 16
model = LSTMAE(input_size, hidden_size, num_layers=1, isCuda=False)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

losses = deque([], maxlen=print_interval)

for epoch in range(num_epochs):
    batch_index = 0
    for batch_i, samples in enumerate(dataloader):
        model.zero_grad()
        model.init_hidden()
        word_set = samples
        batch_index += 1
        print('This is the {}th batch.'.format(batch_index))

        loss = 0.

        for char_ind in range(batch_size):
            output = model(samples[char_ind])
            target = samples[char_ind].clone()
            loss += loss_func(output, target)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

'''
        if batch_i % print_interval == 0:
            print(generate('Charles', dataset, model))
            print('[%03d] %05d/%05d Loss: %.4f' % (
                epoch + 1,
                batch_i,
                len(dataset) // batch_size,
                sum(losses) / len(losses)
            ))
'''

torch.save(model.state_dict(), 'model.pt')
