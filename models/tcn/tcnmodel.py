import torch.nn.functional as F
from torch import nn
from models.tcn.tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.encoder = nn.Linear(input_size, num_channels[0])
        self.tcn = TemporalConvNet(num_channels[0], num_channels[1:], kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.encoder(inputs)
        x = x.transpose(-1, -2) 
        x = self.tcn(x)
        x = x.transpose(-1, -2) 
        x = self.decoder(x)
        return x
    
        # y1 = self.tcn(inputs)  # (N, C, L) -> (N, C, L)
        # o = self.linear(y1[:, :, -1]) # (N, C) -> (N, O)
        # return F.log_softmax(o, dim=1) # (N, O) -> (N, O)
        