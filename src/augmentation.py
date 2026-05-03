"""
Six skeletal-space augmentation operations for (T, 21, 3) keypoint sequences.
All operations preserve anatomical plausibility and background invariance
(no pixel data involved).

Operations
----------
1. MirrorFlip      — negate x-axis  (always applied)
2. InPlaneRotation — random 2-D rotation in xy-plane
3. ScaleJitter     — random uniform scale
4. SpeedJitter     — temporal resampling at random speed
5. GaussianNoise   — independent noise per coordinate
6. MixSkel         — intra-class linear interpolation (Mixup in keypoint space)
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (T,
                    AUG_ROTATION_DEG, AUG_SCALE_RANGE,
                    AUG_SPEED_RANGE,  AUG_NOISE_SIGMA,
                    AUG_MIXSKEL_ALPHA,
                    AUG_PROB_NORMAL,  AUG_PROB_HARD,
                    HARD_CLASSES, IDX_TO_LABEL)
from src.preprocess import resample_sequence


# ── Individual operations ──────────────────────────────────────────────────────

def mirror_flip(seq: np.ndarray) -> np.ndarray:
    """Negate x-coordinate of all keypoints every frame."""
    out = seq.copy()
    out[:, :, 0] *= -1
    return out


def in_plane_rotation(seq: np.ndarray,
                      max_deg: float = AUG_ROTATION_DEG) -> np.ndarray:
    """Random 2-D rotation in the xy-plane, constant across all frames."""
    theta = np.radians(np.random.uniform(-max_deg, max_deg))
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s], [s, c]], dtype=np.float32)   # (2, 2)
    out   = seq.copy()
    out[:, :, :2] = (out[:, :, :2] @ R.T)                  # broadcast (T,21,2)@(2,2)
    return out


def scale_jitter(seq: np.ndarray,
                 lo: float = AUG_SCALE_RANGE[0],
                 hi: float = AUG_SCALE_RANGE[1]) -> np.ndarray:
    """Multiply all coordinates by a single random scale factor."""
    alpha = np.random.uniform(lo, hi)
    return seq * alpha


def speed_jitter(seq: np.ndarray,
                 lo: float = AUG_SPEED_RANGE[0],
                 hi: float = AUG_SPEED_RANGE[1]) -> np.ndarray:
    """
    Stretch or compress the temporal axis by factor beta, then resample
    back to T frames.  Simulates signers holding a pose for different durations.
    """
    beta      = np.random.uniform(lo, hi)
    T_in      = len(seq)
    T_new     = max(3, int(round(T_in * beta)))
    src_idx   = np.linspace(0, T_in - 1, T_new)
    lo_i      = np.floor(src_idx).astype(int)
    hi_i      = np.minimum(lo_i + 1, T_in - 1)
    frac      = (src_idx - lo_i).reshape(-1, 1, 1)
    stretched = seq[lo_i] * (1 - frac) + seq[hi_i] * frac
    return resample_sequence(stretched.astype(np.float32), T)


def gaussian_noise(seq: np.ndarray,
                   sigma: float = AUG_NOISE_SIGMA) -> np.ndarray:
    """Add independent Gaussian noise to every coordinate."""
    noise = np.random.normal(0.0, sigma, seq.shape).astype(np.float32)
    return seq + noise


def mixskel(seq_a: np.ndarray, seq_b: np.ndarray,
            lo: float = AUG_MIXSKEL_ALPHA[0],
            hi: float = AUG_MIXSKEL_ALPHA[1]) -> np.ndarray:
    """
    Intra-class linear interpolation between two same-class sequences.
    seq_a and seq_b must both be (T, 21, 3) after resampling.
    """
    lam = np.random.uniform(lo, hi)
    return (lam * seq_a + (1.0 - lam) * seq_b).astype(np.float32)


# ── Augmenter class ────────────────────────────────────────────────────────────

class SkeletonAugmenter:
    """
    Applies the six augmentation operations to a (T, 21, 3) sequence.

    Parameters
    ----------
    label_idx  : int — class index of the sequence
    class_pool : dict[int, List[np.ndarray]] — mapping from label_idx to
                 all same-class training sequences (needed for MixSkel)
    training   : bool — if False, no augmentation is applied
    """

    def __init__(self,
                 class_pool: Optional[dict] = None,
                 training:   bool = True):
        self.class_pool = class_pool or {}
        self.training   = training

    def _prob(self, label_idx: int) -> float:
        label_str = IDX_TO_LABEL.get(label_idx, "")
        return AUG_PROB_HARD if label_str in HARD_CLASSES else AUG_PROB_NORMAL

    def __call__(self,
                 seq:       np.ndarray,
                 label_idx: int) -> np.ndarray:
        if not self.training:
            return seq

        seq = seq.copy()
        p   = self._prob(label_idx)

        # Op 1: mirror (always)
        if np.random.rand() < 0.5:
            seq = mirror_flip(seq)

        # Op 5: noise (always)
        seq = gaussian_noise(seq)

        # Op 2: rotation
        if np.random.rand() < p:
            seq = in_plane_rotation(seq)

        # Op 3: scale
        if np.random.rand() < p:
            seq = scale_jitter(seq)

        # Op 4: speed jitter
        if np.random.rand() < 0.5:
            seq = speed_jitter(seq)

        # Op 6: MixSkel
        if np.random.rand() < p and label_idx in self.class_pool:
            pool = self.class_pool[label_idx]
            if len(pool) > 1:
                partner = pool[np.random.randint(len(pool))]
                seq = mixskel(seq, partner)

        return seq


# ── Offline expansion (optional, for small datasets) ──────────────────────────

def expand_dataset(X: np.ndarray, y: np.ndarray,
                   multiplier: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Offline augmentation: return (X_aug, y_aug) with ~multiplier× more samples.
    Used only if on-the-fly augmentation via SkeletonAugmenter is not preferred.
    """
    from collections import defaultdict
    pool: dict = defaultdict(list)
    for seq, lbl in zip(X, y):
        pool[int(lbl)].append(seq)

    augmenter = SkeletonAugmenter(class_pool=pool, training=True)

    X_new = [X]
    y_new = [y]
    for _ in range(multiplier - 1):
        batch = np.stack([augmenter(seq, int(lbl)) for seq, lbl in zip(X, y)])
        X_new.append(batch)
        y_new.append(y)

    return np.concatenate(X_new, axis=0), np.concatenate(y_new, axis=0)
