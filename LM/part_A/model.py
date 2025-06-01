import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout, output_size, pad_index=0, emb_dropout=0.2, out_dropout=0.3, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        if dropout:
            # Dropout after embedding layer
            self.emb_dropout = nn.Dropout(emb_dropout)
        # LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        if dropout:
            # Dropout before the linear layer
            self.out_dropout = nn.Dropout(out_dropout)
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        # Apply embedding
        emb = self.embedding(input_sequence)
        # Apply dropout after embedding
        if hasattr(self, 'emb_dropout'):
            emb = self.emb_dropout(emb)
        # Process through LSTM
        lstm_out, _ = self.lstm(emb)
        # Apply dropout before linear layer
        if hasattr(self, 'out_dropout'):
            lstm_out = self.out_dropout(lstm_out)
        # Project to output space
        output = self.output(lstm_out).permute(0,2,1)
        return output