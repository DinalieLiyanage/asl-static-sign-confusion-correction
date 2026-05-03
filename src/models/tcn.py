"""
M2 — Temporal Convolutional Network (TCN)
Dilated causal 1-D convolutions with residual connections.
Input : (B, T, 21, 3)  →  flatten to  (B, 63, T)  (channels-first for Conv1d)
Output: logits (B, num_classes), embeddings (B, channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FEATURE_DIM, NUM_CLASSES, TCN_CHANNELS


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation   # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.norm1   = nn.BatchNorm1d(out_ch)
        self.norm2   = nn.BatchNorm1d(out_ch)
        self.drop    = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv1(x)[:, :, :x.size(2)]   # trim causal padding
        out = self.drop(F.relu(self.norm1(out)))
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.drop(F.relu(self.norm2(out)))
        res = self.downsample(x) if self.downsample else x
        return F.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self,
                 input_dim:   int = FEATURE_DIM,
                 channels:    int = TCN_CHANNELS,
                 kernel_size: int = 3,
                 num_classes: int = NUM_CLASSES,
                 dropout:     float = 0.2):
        super().__init__()
        dilations = [1, 2, 4, 8, 16]
        layers = []
        in_ch  = input_dim
        for d in dilations:
            layers.append(_ResidualBlock(in_ch, channels, kernel_size, d, dropout))
            in_ch = channels
        self.tcn     = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor):
        B, T_, K, C = x.shape
        x   = x.view(B, T_, K * C).permute(0, 2, 1)   # (B, 63, T)
        out = self.tcn(x)                               # (B, channels, T)
        emb = out.mean(dim=-1)                          # global avg pool → (B, channels)
        emb = self.dropout(emb)
        return self.head(emb), emb
