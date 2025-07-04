import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class VariationalDropout(nn.Module):
    def __init__(self, dropout_prob):
        super(VariationalDropout, self).__init__()
        if not 0.0 <= dropout_prob < 1.0:
            raise ValueError("dropout_prob must be in the half‑open interval [0, 1).")
        self.dropout_prob = dropout_prob

    def forward(self, inputs):
        if (not self.training) or self.dropout_prob == 0.0:
            return inputs
        # inputs shape: (seq_len, batch_size, feature_dim)
        # Sample mask once per sequence
        device = inputs.device
        mask = torch.bernoulli(torch.full((inputs.shape[2],), 1 - self.dropout_prob, device=device))
        # Scale to keep expectation
        mask = mask / (1 - self.dropout_prob)
        # Expand mask for all sequence timesteps
        mask = mask.unsqueeze(0).expand(inputs.shape)
        # Apply mask
        return inputs * mask

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, emb_dropout=0.5, out_dropout=0.5, n_layers=1, weight_tying=True, variational_dropout=True):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Choose dropout implementation
        if variational_dropout:
            self.emb_dropout = VariationalDropout(emb_dropout)
            self.out_dropout = VariationalDropout(out_dropout)
        else:
            # Identity = no dropout (probability effectively 0.0)
            self.emb_dropout = nn.Identity()
            self.out_dropout = nn.Identity()
        
        # LSTM
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        self.output = nn.Linear(hidden_size, output_size, bias=False)
         # Weight tying
        if weight_tying:
            if emb_size != hidden_size:
                raise ValueError(
                    "Weight tying requires emb_size == hidden_size (got "
                    f"{emb_size} vs {hidden_size})."
                )
            # Share parameters
            self.output.weight = self.embedding.weight
        else:
            raise ValueError("Weight tying is not possible: embedding size and hidden size must be equal.")
    
    def forward(self, input_sequence):
        emb = self.emb_dropout(self.embedding(input_sequence))
        lstm_out, _ = self.lstm(emb)
        out = self.out_dropout(lstm_out)
        output = self.output(out).permute(0,2,1)
        return output