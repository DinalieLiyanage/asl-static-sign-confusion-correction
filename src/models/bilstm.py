"""
M1 — Bidirectional LSTM
Input : (B, T, 21, 3)  →  flatten to  (B, T, 63)
Output: logits (B, num_classes), embeddings (B, hidden*2)
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import T, FEATURE_DIM, NUM_CLASSES, BILSTM_HIDDEN


class BiLSTMClassifier(nn.Module):
    def __init__(self,
                 input_dim:   int = FEATURE_DIM,
                 hidden_dim:  int = BILSTM_HIDDEN,
                 num_layers:  int = 2,
                 num_classes: int = NUM_CLASSES,
                 dropout:     float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor):
        # x : (B, T, 21, 3)
        B, T_, K, C = x.shape
        x = x.view(B, T_, K * C)                     # (B, T, 63)

        out, (hn, _) = self.lstm(x)                  # out: (B,T, hidden*2)
        # Concatenate last forward and backward hidden states
        emb  = torch.cat([hn[-2], hn[-1]], dim=-1)   # (B, hidden*2)
        emb  = self.dropout(emb)
        return self.head(emb), emb
