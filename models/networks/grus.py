from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class GrusModel(nn.Module):
    ''' Grus model
    '''
    def __init__(self, num_gru, input_sizes, hidden_sizes, num_layers=2, dropout=0.3):
        super(GrusModel, self).__init__()
        self.num_gru = num_gru
        self.input_sizes = input_sizes
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers

        self.rnns = nn.ModuleList([nn.GRU(self.input_sizes[i], self.hidden_sizes[i], self.num_layers, batch_first=True, dropout=dropout) for i in range(self.num_gru)])

        # GRU后还要再加一个dropout。因为nn.GRU中加入的dropout不包含最后一层: "introduces a Dropout layer on the outputs of each GRU layer 'except the last layer'"，
        #   但文章中说For the GRU, we apply dropout of 0.3 in "each layer".
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for i in range(self.num_gru)])
        

    def forward(self, x_list, state_list):
        '''
        Parameters:
        ------------------------
        x_list: input feature seqences list
        state_list: h list
        '''
        r_out_list = []
        hidden_list = []
        for i in range(self.num_gru):
            r_out, hidden = self.rnns[i](x_list[i], state_list[i])
            r_out = self.dropouts[i](r_out)
            r_out_list.append(r_out)
            hidden_list.append(hidden)
        cat_r_out = torch.cat(r_out_list, dim=-1)
        return cat_r_out, hidden_list


if __name__ == '__main__':
    model = GrusModel(2, [2048,128], [512,128], num_layers=2, dropout=0.3)
    input_1 = torch.rand((8, 60, 2048)) #(bs, seq_len, input_size)
    input_2 = torch.rand((8, 60, 128)) #(bs, seq_len, input_size)
    input = [input_1, input_2]
    state_1 = torch.zeros((2, 8, 512)) #(num_layers, bs, hidden_size)
    state_2 = torch.zeros((2, 8, 128)) #(num_layers, bs, hidden_size)
    state = [state_1, state_2]


    out, state = model(input, state)
    print(model)
    print(out.shape, state[0].shape, state[1].shape)