"""
Loss functions used during training.

CombinedLoss = CrossEntropyLoss (label-smoothed) + λ · SupConLoss
SupConLoss is applied only for M4 (ST-GCN) and M5 (GAT-Transformer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_SMOOTH, SUPCON_LAMBDA


# ── Supervised Contrastive Loss ────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Operates on L2-normalised embeddings.

    Parameters
    ----------
    temperature : float  — controls sharpness of the distribution
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, D) float  — L2-normalised embeddings
        labels   : (B,)   long   — class indices

        Returns
        -------
        scalar loss
        """
        B = features.shape[0]
        device = features.device

        # Cosine similarity matrix scaled by temperature
        sim = torch.matmul(features, features.T) / self.temperature   # (B, B)

        # Mask: 1 where labels match, 0 otherwise; zero out diagonal
        labels = labels.unsqueeze(1)                                    # (B,1)
        pos_mask = (labels == labels.T).float().to(device)             # (B, B)
        self_mask = torch.eye(B, device=device)
        pos_mask  = pos_mask - self_mask                               # remove self

        # Log-softmax over all negatives (exclude self)
        exp_sim    = torch.exp(sim) * (1 - self_mask)                  # (B, B)
        log_prob   = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # Mean log-prob of positive pairs
        num_pos    = pos_mask.sum(dim=1).clamp(min=1)
        loss       = -(pos_mask * log_prob).sum(dim=1) / num_pos
        return loss.mean()


# ── Combined Loss ──────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    CE + λ · SupCon

    Parameters
    ----------
    use_supcon : bool — if False, only CE is used (M1, M2, M3)
    lam        : float — weight for SupCon term
    """

    def __init__(self, use_supcon: bool = False,
                 lam: float = SUPCON_LAMBDA,
                 label_smoothing: float = LABEL_SMOOTH):
        super().__init__()
        self.ce       = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.supcon   = SupConLoss() if use_supcon else None
        self.lam      = lam
        self.use_supcon = use_supcon

    def forward(self, logits:    torch.Tensor,
                      labels:    torch.Tensor,
                      embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        logits     : (B, C)  — raw classifier output
        labels     : (B,)    — ground-truth indices
        embeddings : (B, D)  — L2-normalised features (only if use_supcon=True)
        """
        loss = self.ce(logits, labels)

        if self.use_supcon and embeddings is not None:
            emb_norm = F.normalize(embeddings, dim=-1)
            loss = loss + self.lam * self.supcon(emb_norm, labels)

        return loss
