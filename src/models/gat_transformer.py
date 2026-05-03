"""
M5 — Hybrid Graph Attention Transformer
Stage 1 (Spatial): Graph Attention Network (GAT) layers over the 21 hand nodes.
Stage 2 (Temporal): Transformer Encoder over the T time-step graph embeddings.

Input : (B, T, 21, 3)
Output: logits (B, num_classes), embeddings (B, d_model)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (T, NUM_KEYPOINTS, COORDS, NUM_CLASSES,
                    GAT_HEADS, GAT_D_MODEL, TRANS_LAYERS, TRANS_FF_DIM,
                    HAND_EDGES)


def _build_edge_index(num_nodes: int, edges: list) -> torch.Tensor:
    """
    Build (2, E) edge_index tensor (undirected: both directions included).
    """
    src, dst = [], []
    for i, j in edges:
        src += [i, j]
        dst += [j, i]
    # Self-loops
    for i in range(num_nodes):
        src.append(i); dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)   # (2, E)


class _GATLayer(nn.Module):
    """
    Single-head Graph Attention layer (Veličković et al., ICLR 2018).
    Operates on node features of one frame.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W   = nn.Linear(in_dim,  out_dim, bias=False)
        self.att = nn.Linear(out_dim * 2, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        h          : (V, in_dim)
        edge_index : (2, E)
        Returns    : (V, out_dim)
        """
        h  = self.W(h)                                   # (V, out_dim)
        src, dst = edge_index[0], edge_index[1]
        e  = torch.cat([h[src], h[dst]], dim=-1)         # (E, 2*out_dim)
        a  = self.act(self.att(e)).squeeze(-1)           # (E,)

        # Softmax over neighbours
        V     = h.size(0)
        alpha = torch.zeros(V, V, device=h.device)
        alpha[src, dst] = a
        alpha = torch.softmax(alpha, dim=-1)             # (V, V)
        alpha = self.drop(alpha)

        return alpha @ h                                  # (V, out_dim)


class _MultiHeadGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        head_dim   = out_dim // num_heads
        self.heads = nn.ModuleList([
            _GATLayer(in_dim, head_dim, dropout) for _ in range(num_heads)
        ])
        self.norm  = nn.LayerNorm(out_dim)
        # Residual projection if dims differ
        self.res   = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, h: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(h, edge_index) for head in self.heads], dim=-1)
        res = self.res(h) if self.res else h
        return F.elu(self.norm(out + res))


class GATTransformerClassifier(nn.Module):
    def __init__(self,
                 in_channels: int = COORDS,
                 d_model:     int = GAT_D_MODEL,
                 num_heads:   int = GAT_HEADS,
                 num_gat_layers: int = 2,
                 num_nodes:   int = NUM_KEYPOINTS,
                 num_classes: int = NUM_CLASSES,
                 trans_layers: int = TRANS_LAYERS,
                 trans_ff:    int = TRANS_FF_DIM,
                 dropout:     float = 0.1):
        super().__init__()

        # Pre-compute edge index (registered as buffer, device-agnostic)
        edge_index = _build_edge_index(num_nodes, HAND_EDGES)
        self.register_buffer("edge_index", edge_index)   # (2, E)

        # ── Stage 1: Spatial GAT ────────────────────────────────────────────
        gat_layers = []
        in_dim     = in_channels
        for _ in range(num_gat_layers):
            gat_layers.append(_MultiHeadGAT(in_dim, d_model, num_heads, dropout))
            in_dim = d_model
        self.gat_layers = nn.ModuleList(gat_layers)

        # Aggregate node embeddings → single graph embedding per frame
        self.node_agg = nn.Linear(d_model, d_model)

        # ── Stage 2: Temporal Transformer ──────────────────────────────────
        self.cls     = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=trans_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc     = nn.TransformerEncoder(enc_layer, num_layers=trans_layers,
                                              enable_nested_tensor=False)
        self.norm    = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls, std=0.02)

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B*T, 21, 3)  →  graph embeddings  →  (B*T, d_model)
        """
        for gat in self.gat_layers:
            x = gat(x, self.edge_index)          # (B*T, 21, d_model)
        # Mean-pool over nodes
        x = x.mean(dim=1)                        # (B*T, d_model)
        return F.elu(self.node_agg(x))

    def forward(self, x: torch.Tensor):
        B, T_, V, C = x.shape                    # (B, T, 21, 3)

        # Run GAT frame by frame (batch all frames together for efficiency)
        x_flat  = x.view(B * T_, V, C)           # (B*T, 21, 3)
        g_emb   = self._spatial(x_flat)          # (B*T, d_model)
        g_emb   = g_emb.view(B, T_, -1)          # (B, T, d_model)

        # Temporal transformer with CLS token
        cls     = self.cls.expand(B, -1, -1)     # (B, 1, d_model)
        tokens  = torch.cat([cls, g_emb], dim=1) # (B, T+1, d_model)
        tokens  = self.enc(tokens)               # (B, T+1, d_model)

        emb     = self.norm(tokens[:, 0])        # CLS → (B, d_model)
        emb     = self.drop(emb)
        return self.head(emb), emb
