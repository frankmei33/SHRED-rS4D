import torch.nn as nn
import torch
import torch.optim as optim

from models.s4.s4 import S4Block as S4
from models.s4.s4d import S4D

import matplotlib.pyplot as plt
import math
from scipy.signal import bode

# # Dropout broke in PyTorch 1.11
# if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
#     print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
#     dropout_fn = nn.Dropout
# if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
#     dropout_fn = nn.Dropout1d
# else:
#     dropout_fn = nn.Dropout2d
dropout_fn = nn.Dropout


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, perm=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        if perm is not None:
            pe = pe[:, perm, :]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# define a trainable positional encoding
class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# a 2D sinusoid positional encoding
class PositionalEncoding3(nn.Module):
    pass
    def __init__(self, d_model, dropout=0.1, width=32, height=32, perm=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(width*height, d_model)
        position = torch.arange(0, width*height, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 4).float() * (-math.log(10000.0) / d_model))
        pe[:, 0:d_model/2:2] = torch.sin((position % width) * div_term)
        pe[:, 1:d_model/2+1:2] = torch.cos((position % width) * div_term)
        pe[:, d_model/2::2] = torch.sin((position // width) * div_term)
        pe[:, d_model/2+1::2] = torch.cos((position // width) * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        if perm is not None:
            pe = pe[:, perm, :]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        n_decoder=1,
        dropout=0.2,
        prenorm=False,
        normalization='layernorm',
        d_filter=0,
        pos_encode=None,
        s4_dropout=0.0,
        s4_lr=None,
        act_fn='sigmoid',
        **kwargs,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.d_filter = d_filter
        self.mode = kwargs['mode']

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Positional encoding
        self.pos_encoder = None
        if pos_encode == 'fixed':
            self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
        elif pos_encode == 'trainable':
            self.pos_encoder = PositionalEncoding2(d_model, dropout=0.1)

        # normalization
        self.normalization = normalization
        if self.normalization == 'layernorm':
            norm_fn = nn.LayerNorm
        elif self.normalization == 'batchnorm':
            norm_fn = nn.BatchNorm1d

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # add S4 filter layer
        if self.d_filter:
            self.s4_layers.append(
                    S4(d_model, mode='diag', d_state=self.d_filter, dropout=s4_dropout, transposed=True, 
                    lr=s4_lr, init='filter')
                )
            self.norms.append(norm_fn(d_model))
            self.dropouts.append(dropout_fn(dropout))

        for _ in range(n_layers):
            self.s4_layers.append(
                # S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
                S4(d_model, dropout=s4_dropout, lr=s4_lr, **kwargs)
                # S4(d_model, mode='diag', dropout=dropout, transposed=True, 
                #     init='diag-lin', bidirectional=False, disc='zoh', real_transform='exp', lr={'B':0.0})
            )
            self.norms.append(norm_fn(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # decoder
        # self.decoder1 = nn.Linear(d_model, d_model*2)
        # self.decoder2 = nn.Linear(d_model*2, d_output)
        # self.dc_dropout = dropout_fn(dropout)
        self.decoder_ln = nn.ModuleList()
        self.decoder_dp = nn.ModuleList()
        d_in, d_out = d_model, d_model*2
        for _ in range(n_decoder):
            self.decoder_ln.append(nn.Linear(d_in, d_out))
            self.decoder_dp.append(dropout_fn(dropout))
            d_in, d_out = d_out, d_out*2
        self.out = nn.Linear(d_in, d_output)

        if act_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif act_fn == 'glu':
            self.activation = nn.GELU()

        print(s4_dropout, s4_lr, kwargs)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        
        # add positional encoding to x
        if self.pos_encoder:
            x = self.pos_encoder(x)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            # Prenorm
            if self.prenorm:
                if self.normalization == 'layernorm':
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)
                elif self.normalization == 'batchnorm':
                    z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            # Postnorm
            if not self.prenorm:
                if self.normalization == 'layernorm':
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                elif self.normalization == 'batchnorm':
                    x = norm(x)

        x = x.transpose(-1, -2) # (B, d_model, L) -> (B, L, d_model)

        # Decode the outputs
        # x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)
        # x = self.decoder1(x)
        # x = self.dc_dropout(x)
        # x = self.activation(x)
        # x = self.decoder2(x)

        for layer, dropout in zip(self.decoder_ln, self.decoder_dp):
            x = layer(x)
            x = dropout(x)
            self.activation(x)
        x = self.out(x)

        return x

    # def update_C(self):
    #     for layer in self.s4_layers:
    #         layer.update_C()

    def energy_loss(self):
        if self.mode not in ['diag', 's4d']:
            return 0

        loss = 0
        layers = self.s4_layers[1:] if self.d_filter>0 else self.s4_layers
        for layer in layers:
            dt, A, B, C = layer.layer.kernel._get_params()

            M = -1/(A[:,:,None].conj() + A[:,None,:]) # (H N N)
            X = B * C
            loss += torch.einsum('iaj,ajk,iak',X.conj(),M,X)/M.shape[0]
        return loss.real / len(layers)

    def dynamics_plot(self, layer_i):
        if self.mode in ['diag', 's4d']:
            layer = self.s4_layers[layer_i]
            dt, A, B, C = layer.layer.kernel._get_params()
            dtA = A.flatten().detach().cpu().numpy()
            # dtA = (dt * A).flatten().detach()
            plt.scatter(dtA.real, dtA.imag, alpha=.5)#, label='Layer {}'.format(layer_i))
            # plt.legend()
        elif self.mode in ['s4']:
            layer = self.s4_layers[layer_i]
            dt, A, B, C, P, Q = layer.layer.kernel._get_params()
            A = torch.diag_embed(A)-torch.einsum('ij,ik->ijk',P,Q)
            A = torch.linalg.eigvals(A)
            # dtA = (dt * A).flatten().detach()
            dtA = A.flatten().detach().cpu().numpy()
            plt.scatter(dtA.real, dtA.imag, alpha=.5)#, label='Layer {}'.format(layer_i))
            # plt.legend()
        else: raise NotImplementedError
    
    # bode plots for SSM on layer i dim j
    def bode_plot(self, layer_i, j):
        assert self.d_filter
        if self.mode in ['diag', 's4d']:
            layer = self.s4_layers[layer_i]
            dt, A, B, C = layer.layer.kernel._get_params()
        elif self.mode in ['s4']:
            layer = self.s4_layers[layer_i]
            dt, A, B, C, _, _ = layer.layer.kernel._get_params()
            A = torch.diag_embed(A)-torch.einsum('ij,ik->ijk',P,Q)
            A = torch.linalg.eigvals(A)
        A = A[j].detach().cpu()
        B = B[0,j,:,None].detach().cpu()
        C = C[:,j,:].detach().cpu()
        A = torch.diag(torch.cat((A,A.conj()))).type(torch.cdouble)
        B = torch.cat((B,B.conj())).type(torch.cdouble)
        C = torch.cat((C,C.conj()), dim=1).type(torch.cdouble)
        D = torch.zeros(C.shape[0],1,device=C.device).type(torch.cdouble)
        # print(A.type())
        # print(A.shape,B.shape,C.shape,D.shape)
        sample_rate = math.log10(1.0/dt[j].detach().cpu()/2)
        w, mag, phase = bode((A,B,C,D),w=torch.logspace(-1,sample_rate,100))

        plt.subplot(2,1,1)
        plt.semilogx(w, mag)    # Bode magnitude plot
        plt.tick_params(labelbottom=False)
        plt.ylabel('Magnitude')
        plt.subplot(2,1,2)
        plt.semilogx(w, phase)  # Bode phase plot
        plt.xlabel('Frequency')
        plt.ylabel('Phase')

def setup_optimizer(model, lr, weight_decay, epochs, patience=0):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    if patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

