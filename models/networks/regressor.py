import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class FcRegressor(nn.Module):
    def __init__(self, input_dim, layers, output_dim, activation=nn.ReLU(), dropout=0.3, use_bn=False, dropout_input=True):
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
            self.all_layers.append(activation)
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        
        self.module = nn.Sequential(*self.all_layers) #前面的所有隐藏层
        self.fc_out = nn.Linear(layers[-1], output_dim) #最后一层输出层
    
    def forward(self, x):
        feat = self.module(x)
        out = self.fc_out(feat)
        return out, feat


if __name__ == '__main__':
    model = FcRegressor(2048, (128,), 15, dropout=0, dropout_input=False)
    input = torch.rand((8, 60, 2048)) # (bs, seq_len, embd_dim)
    output, ft = model(input)
    print(model)
    print(output.shape) # (8, 60, 15)