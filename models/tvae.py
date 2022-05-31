import numpy as np
import torch
import torch.nn as nn


class TVAE(nn.Module):
    def __init__(self, state_dim, z_dim, h_dim, rnn_dim, num_layers):
        super(TVAE, self).__init__()

        # encoder model
        self.enc_rnn = nn.GRU(state_dim, rnn_dim,
                              num_layers, bidirectional=True)
        self.enc_fc = nn.Sequential(
            nn.Linear(2 * rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_logvar = nn.Linear(h_dim, z_dim)
        # TODO: decoder model
