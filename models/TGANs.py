import torch.nn as nn
import torch
import numpy as np


class T2V(nn.Module):
    def __init__(self, input_size=None, output_dim=None):
        super(T2V, self).__init__()

        self.linear = nn.Linear(input_size, output_dim)

    def forward(self, x):
        out = self.linear(x)
        out = torch.sine(out)
        return out


class T_encoder(nn.Module):
    def __init__(self, num_features, seq_length, pred_length):
        super(T_encoder, self).__init__()

        self.num_feats = num_features
        self.seq_len = seq_length
        self.pred_length = pred_length

        self.transf1 = nn.TransformerEncoderLayer(d_model=num_features, nhead=8, activation='gelu')

    def forward(self, x):
        pass
