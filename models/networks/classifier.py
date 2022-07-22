import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, fc1_size, output_size, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.rnn1 = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm((hidden_size * 2, ))
        self.bn = nn.BatchNorm1d(hidden_size * 4)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

    def rnn_flow(self, x, lengths):
        batch_size = lengths.size(0)
        h1, h2 = self.extract_features(x, lengths, self.rnn1, self.rnn2, self.layer_norm)
        h = torch.cat((h1, h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def mask2length(self, mask):
        ''' mask [batch_size, seq_length, feat_size]
        '''
        _mask = torch.mean(mask, dim=-1).long() # [batch_size, seq_len]
        length = torch.sum(_mask, dim=-1)       # [batch_size,]
        return length 

    def forward(self, x, mask):
        lengths = self.mask2length(mask)
        h = self.rnn_flow(x, lengths)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o, h

class SimpleClassifier(nn.Module):
    ''' Linear classifier, use embedding as input
        Linear approximation, should append with softmax
    '''
    def __init__(self, embd_size, output_dim, dropout):
        super(SimpleClassifier, self).__init__()
        self.dropout = dropout
        self.C = nn.Linear(embd_size, output_dim)
        self.dropout_op = nn.Dropout(dropout)

    def forward(self, x):
        if self.dropout > 0:
            x = self.dropout_op(x)
        return self.C(x)
    
class FcClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim, activation=nn.ReLU, dropout=0.3, use_bn=False, dropout_input=True):
        ''' Fully Connect classifier
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            output_dim: output feature dim
            activation: activation function
            dropout: dropout rate
            dropout_input: dropout operation on input feature
        '''
        super().__init__()
        self.all_layers = [nn.Dropout(dropout)] if dropout_input else []
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        
        self.module = nn.Sequential(*self.all_layers)
        self.fc_out = nn.Linear(layers[-1], output_dim)
    
    def forward(self, x):
        feat = self.module(x)
        out = self.fc_out(feat)
        return out, feat

class MaxPoolFc(nn.Module):
    def __init__(self, hidden_size, num_class=4):
        super(MaxPoolFc, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_class),
            nn.ReLU()
        )
    
    def forward(self, x):
        ''' x shape => [batch_size, seq_len, hidden_size]
        '''
        batch_size, seq_len, hidden_size = x.size()
        x = x.view(batch_size, hidden_size, seq_len)
        # print(x.size())
        out = torch.max_pool1d(x, kernel_size=seq_len)
        out = out.squeeze()
        out = self.fc(out)
        
        return out

if __name__ == '__main__':
    a = FcClassifier(256, [500], 4, dropout=0.4, dropout_input=False)
    print(a)
