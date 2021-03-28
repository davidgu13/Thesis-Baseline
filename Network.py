import torch.nn as nn
import torch.nn.functional as F
from preprocessing import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
# seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)

# class BasicNetwork(nn.Module):
#     # Might not be used...
#     def __init__(self, input_size, embed_size, hid_dim, output_size):
#         super(BasicNetwork, self).__init__()
#         self.input_size = input_size
#         self.embed_size = embed_size
#         self.embedding = nn.Embedding(input_size, embed_size)
#         # Encoder
#         self.encoder = nn.LSTM(embed_size, hid_dim, bidirectional=IS_BIDIRECTIONAL)
#
#         # Decoder
#         self.decoder = nn.LSTM(2*hid_dim, output_size)
#
#         # self.softmax = torch.softmax()
#
#     def forward(self, inp):
#         embed = self.embedding(inp)
#         h, x = self.encoder(embed)
#         return x, h

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hid_dim):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hid_dim
        self.hid_state_size = self.hidden_size * (1 + int(IS_BIDIRECTIONAL))
        self.embedding = nn.Embedding(input_size, hid_dim)

        self.lstm = nn.LSTM(hid_dim, hid_dim, bidirectional=IS_BIDIRECTIONAL)

    def forward(self, input_padded, input_lens, hidden:tensor):
        # inp, lens = pad_packed_sequence(inp, padding_value=PAD_token)
        inp_embedded = self.embedding(input_padded)
        inp_packed = pack_padded_sequence(inp_embedded,torch.tensor(input_lens),enforce_sorted=False)

        # embedded = embedded.view(1, 1, -1)
        # output = inp_embedded #.unsqueeze(0)
        output_packed, hidden = self.lstm(inp_packed, hidden)
        output_padded, output_lengths = pad_packed_sequence(output_packed)

        return output_padded, output_lengths, hidden


    def initHidden(self, batch_size):
        h0 = torch.randn(1, batch_size, self.hid_state_size, device=device) # .view(-1, 1, self.hidden_size)
        c0 = torch.randn(1, batch_size, self.hid_state_size, device=device) # .view(-1, 1, self.hidden_size)
        return h0, c0


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.hid_state_size = self.hidden_size * (1 + int(IS_BIDIRECTIONAL))

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp, hidden):
        t = self.embedding(inp).view(1, 1, -1)
        t = F.relu(t)
        output, hidden = self.lstm(t, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        h0 = torch.randn(1, 1, self.hid_state_size, device=device).view(-1, 1, self.hidden_size)
        c0 = torch.randn(1, 1, self.hid_state_size, device=device).view(-1, 1, self.hidden_size)
        return h0, c0


class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_OUT_LENGTH):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.hid_state_size = self.hidden_size * (1 + int(IS_BIDIRECTIONAL))
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, inp, hidden, encoder_outputs):
        batch_size = inp.shape[1]
        embedded = self.embedding(inp) #.view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax( self.attn( torch.cat((embedded[0],hidden[0][0]),-1) ), dim=-1).t() # add [0] bc it's LSTM and not GRU

        attn_weights = torch.reshape(attn_weights.unsqueeze(0), (batch_size, 1, -1))
        encoder_outputs = torch.reshape(encoder_outputs.unsqueeze(0), (batch_size, self.hidden_size, 1))
        attn_applied = torch.reshape(torch.bmm(encoder_outputs, attn_weights), (-1, batch_size, self.hidden_size))

        output = torch.cat((embedded, attn_applied), dim=2) # removed [0] in both tensors, replaced 1 with 0.
        output = self.attn_combine(output).unsqueeze(0)


        # inp = inp[:,0]
        # hidden = [*hidden]
        # hidden[0] = hidden[0][:,0].unsqueeze(1)
        # hidden[1] = hidden[1][:,0].unsqueeze(1)
        # encoder_outputs = encoder_outputs[0]
        #
        # embedded = self.embedding(inp).view(1, 1, -1)
        # embedded = self.dropout(embedded)
        #
        # attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        #
        # attn_weights = torch.reshape(attn_weights, (1, -1, 1)) #.squeeze(0).unsqueeze(1).unsqueeze(1)
        # encoder_outputs = torch.reshape(encoder_outputs, (1, 1, -1)) #.unsqueeze(0).unsqueeze(2)
        # attn_applied = torch.bmm(attn_weights, encoder_outputs)
        #
        # output = torch.cat((embedded[0], attn_applied[0]), 1)
        # output = self.attn_combine(output).unsqueeze(0)


        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        h0 = torch.randn(1, 1, self.hid_state_size, device=device) #.view(-1, 1, self.hidden_size)
        c0 = torch.randn(1, 1, self.hid_state_size, device=device) #.view(-1, 1, self.hidden_size)
        return h0, c0
