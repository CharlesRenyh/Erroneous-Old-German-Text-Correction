from collections import deque
import string
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import TextDataset
from model import LSTMAE


def generate(word, dataset, model):
    new_word = []
    model.eval()

    # t_word = dataset.to_one_hot(word)
    t_word = torch.LongTensor(word).view(1,1)
    for _ in range(dataset.max_length):
        t_char = model(t_word)[:, -1, :]

        char_idx = t_char.argmax(dim=1)
        new_char = dataset.charset[char_idx.item()]

        if new_char == '\0':
            break
        else:
            new_word.append(new_char)
            # t_char = dataset.to_one_hot(new_char)
            t_word = torch.cat([t_word, char_idx.view(1, 1)], dim=1)
    return new_word


if __name__ == '__main__':
    data_root = '/Users/charlesren/Downloads/AI_lab_data'
    max_length = 6

    # Prepare GPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset.
    dataset = TextDataset(data_root, max_length)

    input_size = 68
    hidden_size = 16

    # Prepare model.
    model = LSTMAE(input_size, hidden_size, 1, False)
    model.load_state_dict(torch.load('model.pt'))
    model = model.to(device)

    new_words = []

    for letter in 'Metallen':
        word = generate(letter, dataset, model)
        print(word)
        new_words.append(word)

