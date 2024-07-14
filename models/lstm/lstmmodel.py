import torch.nn.functional as F
from torch import nn

class LSTM(nn.Module):
    def __init__(self, d_input, d_output=10, d_model=256, n_layers=1, n_decoder=1, dropout=0, act_fn='sigmoid'):
        super().__init__()
        self.projection = nn.Linear(d_input, d_model)
        self.model = nn.LSTM(d_model, d_model, n_layers, dropout=dropout, batch_first=True)
        # self.linear1 = nn.Linear(d_model, d_model*2)
        # self.linear2 = nn.Linear(d_model*2, d_output)
        # self.dropout = nn.Dropout(dropout)

        self.decoder_ln = nn.ModuleList()
        self.decoder_dp = nn.ModuleList()
        d_in, d_out = d_model, d_model*2
        for _ in range(n_decoder):
            self.decoder_ln.append(nn.Linear(d_in, d_out))
            self.decoder_dp.append(nn.Dropout(dropout))
            d_in, d_out = d_out, d_out*2
        self.out = nn.Linear(d_in, d_output)

        if act_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif act_fn == 'glu':
            self.activation = nn.GELU()

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.projection(inputs)
        
        x, _ = self.model(x)
        # x = self.linear1(x)
        # x = self.dropout(x)
        # x = self.activation(x)
        # x = self.linear2(x)

        for layer, dropout in zip(self.decoder_ln, self.decoder_dp):
            x = layer(x)
            x = dropout(x)
            self.activation(x)
        x = self.out(x)

        return x
    
