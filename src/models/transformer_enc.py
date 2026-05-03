"""
M3 — Transformer Encoder
Each frame is a token.  A [CLS] token aggregates the sequence.
Input : (B, T, 21, 3)  →  linear projection  →  (B, T+1, d_model)
Output: logits (B, num_classes), embeddings (B, d_model)
"""

import math
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (T, FEATURE_DIM, NUM_CLASSES,
                    TRANS_D_MODEL, TRANS_HEADS, TRANS_LAYERS, TRANS_FF_DIM)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 input_dim:   int = FEATURE_DIM,
                 d_model:     int = TRANS_D_MODEL,
                 nhead:       int = TRANS_HEADS,
                 num_layers:  int = TRANS_LAYERS,
                 ff_dim:      int = TRANS_FF_DIM,
                 num_classes: int = NUM_CLASSES,
                 dropout:     float = 0.1):
        super().__init__()
        self.proj  = nn.Linear(input_dim, d_model)
        self.cls   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos   = _PositionalEncoding(d_model, max_len=T + 2, dropout=dropout)

        enc_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc   = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                            enable_nested_tensor=False)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T_, K, C = x.shape
        x   = x.view(B, T_, K * C)                      # (B, T, 63)
        x   = self.proj(x)                               # (B, T, d_model)

        cls = self.cls.expand(B, -1, -1)                 # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                 # (B, T+1, d_model)
        x   = self.pos(x)
        x   = self.enc(x)                                # (B, T+1, d_model)
        emb = self.norm(x[:, 0])                         # CLS token → (B, d_model)
        emb = self.drop(emb)
        return self.head(emb), emb
