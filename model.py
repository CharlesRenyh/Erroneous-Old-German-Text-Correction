import torch
import torch.nn as nn

embedding_dim = 16
encode_embedding = nn.Embedding(62, embedding_dim)
decode_embedding = nn.Embedding(embedding_dim, 62)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = encode_embedding
        self.isCuda = isCuda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, input):
        tt = torch.cuda if self.isCuda else torch
        h0 = tt.randn(self.num_layers, input.size(0), self.hidden_size)
        c0 = tt.randn(self.num_layers, input.size(0), self.hidden_size)
        after_encoding, hidden = self.lstm(input, (h0, c0))
        encoded_content = self.relu(after_encoding)
        return encoded_content


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = decode_embedding
        self.isCuda = isCuda
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        h0 = tt.randn(self.num_layers, encoded_input.size(0), self.output_size)
        c0 = tt.randn(self.num_layers, encoded_input.size(0), self.output_size)
        decoded_output, hidden = self.lstm(encoded_input, (h0, c0))
        decoded_output = self.softmax(decoded_output)
        return decoded_output


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, isCuda)

    def forward(self, input):
        hidden_state = self.encoder(input)
        decoded_output = self.decoder(hidden_state)
        return decoded_output


if __name__ == '__main__':
    model = LSTMAE(100, 10, 2, False)
    print(model)

