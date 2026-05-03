"""
M4 — Spatial-Temporal Graph Convolutional Network (ST-GCN)
Implemented from scratch (no PyTorch Geometric required).

Each frame is a 21-node graph with hand skeleton edges.
ST-GCN interleaves spatial graph convolution and temporal convolution.

Input : (B, T, 21, 3)
Output: logits (B, num_classes), embeddings (B, out_channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (T, NUM_KEYPOINTS, COORDS, NUM_CLASSES,
                    STGCN_CHANNELS, HAND_EDGES)


def _build_adj(num_nodes: int, edges: list) -> torch.Tensor:
    """
    Build normalised symmetric adjacency matrix (with self-loops).
    Returns (num_nodes, num_nodes) float tensor.
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)                   # self-loops
    D = np.diag(A.sum(axis=1) ** -0.5)
    A_norm = D @ A @ D
    return torch.from_numpy(A_norm)            # (V, V)


class _STGCNBlock(nn.Module):
    """One ST-GCN block: spatial graph conv → temporal conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int,
                 num_nodes: int, kernel_t: int = 9,
                 dropout: float = 0.1):
        super().__init__()
        self.gcn_w    = nn.Linear(in_ch, out_ch, bias=False)
        pad_t         = (kernel_t - 1) // 2
        self.tcn      = nn.Conv2d(out_ch, out_ch,
                                  kernel_size=(kernel_t, 1),
                                  padding=(pad_t, 0))
        self.bn       = nn.BatchNorm2d(out_ch)
        self.dropout  = nn.Dropout(dropout)
        self.residual = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else None)

    def forward(self, x: torch.Tensor,
                A: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_ch, T, V)
        A : (V, V)
        Returns (B, out_ch, T, V)
        """
        # Spatial graph convolution:  H = A * X * W
        # x.permute → (B, T, V, in_ch)  →  linear →  (B, T, V, out_ch)
        Bsz, _, T_, V = x.shape
        h = x.permute(0, 2, 3, 1)                     # (B, T, V, in_ch)
        h = torch.einsum("ij,btjc->btic", A, h)       # (B, T, V, in_ch) — graph mix
        h = self.gcn_w(h)                              # (B, T, V, out_ch)
        h = h.permute(0, 3, 1, 2)                     # (B, out_ch, T, V)

        # Temporal convolution
        h = self.tcn(h)                                # (B, out_ch, T, V)

        # Residual
        res = self.residual(x) if self.residual else x
        h   = F.relu(self.bn(h) + res)
        return self.dropout(h)


class STGCNClassifier(nn.Module):
    def __init__(self,
                 in_channels: int = COORDS,
                 channels:    list = STGCN_CHANNELS,
                 num_nodes:   int = NUM_KEYPOINTS,
                 num_classes: int = NUM_CLASSES,
                 dropout:     float = 0.1):
        super().__init__()
        A = _build_adj(num_nodes, HAND_EDGES)
        self.register_buffer("A", A)                   # (V, V)

        blocks = []
        in_ch  = in_channels
        for out_ch in channels:
            blocks.append(_STGCNBlock(in_ch, out_ch, num_nodes,
                                       dropout=dropout))
            in_ch = out_ch
        self.blocks   = nn.ModuleList(blocks)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.drop     = nn.Dropout(dropout)
        self.head     = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor):
        # x : (B, T, 21, 3)
        x = x.permute(0, 3, 1, 2)                     # (B, 3, T, 21)

        for block in self.blocks:
            x = block(x, self.A)                       # (B, C, T, 21)

        emb = self.pool(x).squeeze(-1).squeeze(-1)     # (B, C)
        emb = self.drop(emb)
        return self.head(emb), emb
