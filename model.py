import torch
import torch.nn as nn

'''
embedding_dim = 16
encode_embedding = nn.Embedding(50, embedding_dim)
decode_embedding = nn.Embedding(embedding_dim, 50)
'''


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.embedding = encode_embedding
        self.isCuda = isCuda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size), torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, input):
        tt = torch.cuda if self.isCuda else torch
        #embeds = self.embedding(input)
        #h0 = tt.randn(self.num_layers, input.size(0), self.hidden_size)
        #c0 = tt.randn(self.num_layers, input.size(0), self.hidden_size)
        after_encoding, hidden = self.lstm(input.view(1,1,-1), self.hidden)
        encoded_content = self.relu(after_encoding)
        return encoded_content


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        #self.embedding = nn.Embedding(output_size, hidden_size)
        self.num_layers = num_layers
        self.isCuda = isCuda
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.dense = nn.Linear(output_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.output_size), torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        #h0 = tt.randn(self.num_layers, encoded_input.size(0), self.output_size)
        #c0 = tt.randn(self.num_layers, encoded_input.size(0), self.output_size)
        #encoded_input = self.embedding(encoded_input).view(1,1,-1)
        decoded_output, hidden = self.lstm(encoded_input, self.hidden)
        decoded_output = self.dense(decoded_output)
        decoded_output = self.softmax(decoded_output)
        return decoded_output


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.en_hidden = self.init_enhidden()
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, isCuda)
        self.de_hidden = self.init_dehidden()

    def init_enhidden(self):
        return self.encoder.init_hidden()

    def init_dehidden(self):
        return self.decoder.init_hidden()

    def forward(self, input):
        hidden_state = self.encoder(input, (self.en_hidden))
        decoded_output = self.decoder(hidden_state, (self.de_hidden))
        return decoded_output


if __name__ == '__main__':
    model = LSTMAE(100, 10, 2, False)
    print(model)

