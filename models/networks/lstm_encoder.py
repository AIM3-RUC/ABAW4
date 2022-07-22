import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .fc_encoder import FcEncoder

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                            bidirectional=True, num_layers=1)
    
    def forward(self, x, states=None):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        states: (h_0, c_0)
        '''
        r_out, (h_n, h_c) = self.rnn(x, states)
        return r_out, (h_n, h_c)

class AttentiveLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentiveLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                            bidirectional=True, num_layers=1)
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.se = nn.Sequential(
                nn.Conv1d(hidden_size*2, hidden_size // 2, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size // 2, hidden_size // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size // 2, hidden_size*2, kernel_size=1),
                nn.Sigmoid()
        )
        
        self.out_cnn = nn.Sequential(
                nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
        )
    
    def forward(self, x, states=None):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        states: (h_0, c_0)
        '''
        r_out, (h_n, h_c) = self.rnn(x, states)
        # attn = self.se(r_out.transpose(1, 2))
        # attn = attn.transpose(1, 2)
        # return r_out * attn, (h_n, h_c)
        return r_out, (h_n, h_c)

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                       num_layers=1)
    
    def forward(self, x, states):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        states: (h_0, c_0)
        '''
        r_out, (h_n, h_c) = self.rnn(x, states)
        return r_out, (h_n, h_c)

class BiLSTM_official_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                            bidirectional=True, num_layers=1)
    
    def forward(self, x):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        return r_out, (h_n, h_c)

class LSTM_official_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_official_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                       num_layers=1)
    
    def forward(self, x):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        return r_out, (h_n, h_c)

class FcLstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(FcLstmEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = FcEncoder(input_size, [hidden_size, hidden_size], dropout=0.1, dropout_input=False)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True,
                       num_layers=1, bidirectional=bidirectional)
    
    def forward(self, x, states):
        x = self.fc(x)
        r_out, (h_n, h_c) = self.rnn(x, states)
        return r_out, (h_n, h_c)

class AttentionFusionNet(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, hidden_size):
        super(AttentionFusionNet, self).__init__()
        self.a_dim = a_dim
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.hidden_size = hidden_size
        self.mapping = nn.Linear(self.hidden_size, self.hidden_size)
        self.modality_context = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.modality_context.data.normal_(0, 0.05)
        self.A_conv = nn.Conv1d(a_dim, hidden_size, kernel_size=1, padding=0)
        self.V_conv = nn.Conv1d(v_dim, hidden_size, kernel_size=1, padding=0)
        self.L_conv = nn.Conv1d(l_dim, hidden_size, kernel_size=1, padding=0)
        self.rnn = self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, )
    
    def atten_embd(self, a_input, v_input, l_input):
        a_input = a_input.unsqueeze(-2) # [batch_size, seq_len, 1, embd_dim]
        v_input = v_input.unsqueeze(-2)
        l_input = l_input.unsqueeze(-2)
        data = torch.cat([a_input, v_input, l_input], dim=-2) # [batch_size, seq_len, 3, embd_dim]
        batch_size, seq_len, _, embd_dim = data.size()
        proj_data = torch.tanh(self.mapping(data))   # [batch_size, seq_len, 3, hidden_size]
        weight = F.softmax(data @ self.modality_context, dim=-2) # [batch_size, seq_len, 3, 1]
        fusion = torch.sum(data * weight, dim=-2)
        return fusion

    def forward(self, a_input, v_input, l_input, states):
        '''
        Input size [batch_size, seq_len, embd_dim]
        '''
        a_input = self.A_conv(a_input.transpose(1, 2)).permute(0, 2, 1)
        v_input = self.V_conv(v_input.transpose(1, 2)).permute(0, 2, 1)
        l_input = self.L_conv(l_input.transpose(1, 2)).permute(0, 2, 1)
        fusion = self.atten_embd(a_input, v_input, l_input) # [batch_size, seq_len, embd_dim]
        r_out, (h_n, h_c) = self.rnn(fusion, states)
        return r_out, (h_n, h_c)

class AttentionFusionNet2(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, hidden_size):
        super(AttentionFusionNet2, self).__init__()
        self.a_dim = a_dim
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.hidden_size = hidden_size
        self.mapping = nn.Linear(self.hidden_size, self.hidden_size)
        self.A_conv = nn.Conv1d(a_dim, hidden_size, kernel_size=1, padding=0)
        self.V_conv = nn.Conv1d(v_dim, hidden_size, kernel_size=1, padding=0)
        self.L_conv = nn.Conv1d(l_dim, hidden_size, kernel_size=1, padding=0)
        self.context_proj = nn.Linear(3 * hidden_size, hidden_size)
        self.rnn = self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, )
    
    def atten_embd(self, a_input, v_input, l_input):
        batch_size, seq_len, embd_dim = a_input.size()
        context = torch.cat([a_input, v_input, l_input], dim=-1)
        context = torch.tanh(self.context_proj(context)).view(-1, self.hidden_size, 1) # [batch_size * seq_len, hidden_size, 1]
        _a_input = a_input.contiguous().view(batch_size * seq_len, 1, self.hidden_size) # [batch_size * seq_len, 1, hidden_size]
        _v_input = v_input.contiguous().view(batch_size * seq_len, 1, self.hidden_size) # [batch_size * seq_len, 1, hidden_size]
        _l_input = l_input.contiguous().view(batch_size * seq_len, 1, self.hidden_size) # [batch_size * seq_len, 1, hidden_size]
        a_weight = torch.bmm(_a_input, context).view(batch_size, -1, 1)          # [batch_size, seq_len, 1]
        v_weight = torch.bmm(_v_input, context).view(batch_size, -1, 1)
        l_weight = torch.bmm(_l_input, context).view(batch_size, -1, 1)
        weight = torch.cat([a_weight, v_weight, l_weight], dim=-1) # [batch_size, seq_len, 3]
        weight = F.softmax(weight, dim=-1).unsqueeze(-1)
        data = torch.cat([a_input.unsqueeze(-2), v_input.unsqueeze(-2), l_input.unsqueeze(-2)], dim=-2)
        fusion = torch.sum(data * weight, dim=-2)
        return fusion

    def forward(self, a_input, v_input, l_input, states):
        '''
        Input size [batch_size, seq_len, embd_dim]
        '''
        a_input = self.A_conv(a_input.transpose(1, 2)).permute(0, 2, 1)
        v_input = self.V_conv(v_input.transpose(1, 2)).permute(0, 2, 1)
        l_input = self.L_conv(l_input.transpose(1, 2)).permute(0, 2, 1)
        fusion = self.atten_embd(a_input, v_input, l_input) # [batch_size, seq_len, embd_dim]
        r_out, (h_n, h_c) = self.rnn(fusion, states)
        return r_out, (h_n, h_c)


if __name__ == '__main__':
    model = AttentiveLSTMEncoder(345, 256)
    input = torch.rand(32, 300, 345)
    out, _ = model(input)
    print(out.shape)