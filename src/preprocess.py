"""
Normalise raw hand keypoint frames from the Cabana image dataset and build
the processed train / val / test splits saved as compressed .npz files.

Pipeline per image
------------------
1. Translate  — subtract wrist (landmark 0) so wrist is at origin
2. Scale      — divide by max inter-keypoint distance → unit-bounded space
3. Mirror     — negate x for left-hand detections → canonical right-hand frame
4. Replicate  — repeat the single normalised frame T times to produce a
                pseudo-sequence (T, 21, 3) compatible with all temporal models

.npz contents
-------------
X       : (N, T, 21, 3) float32 — normalised pseudo-sequences
X_feat  : (N, 55)       float32 — derived geometric features (angles,
                                   distances, curl ratios, etc.)
y       : (N,)          int64   — class indices
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROC, T, VAL_SPLIT, SEED


# ── Per-frame normalisation ───────────────────────────────────────────────────

def normalize_frame(frame: np.ndarray, is_left: bool = False) -> np.ndarray:
    """
    Normalise a single-frame hand keypoint array.

    Parameters
    ----------
    frame   : (21, 3) float32 — raw MediaPipe landmarks
    is_left : bool — mirror x to produce canonical right-hand coordinates

    Returns
    -------
    (21, 3) float32
    """
    lm = frame.copy().astype(np.float32)

    # Translate: wrist (index 0) → origin
    lm = lm - lm[0:1, :]

    # Scale: divide by max pairwise inter-landmark distance
    diffs = lm[:, None, :] - lm[None, :, :]   # (21, 21, 3)
    d_max = np.linalg.norm(diffs, axis=-1).max()
    if d_max > 1e-6:
        lm = lm / d_max

    # Mirror left hand to canonical right-hand space
    if is_left:
        lm[:, 0] = -lm[:, 0]

    return lm


def resample_sequence(seq: np.ndarray, target_len: int = T) -> np.ndarray:
    """
    Resample a variable-length (T_in, 21, 3) sequence to exactly target_len
    frames via linear interpolation. Used by speed_jitter augmentation.
    """
    T_in = len(seq)
    if T_in == target_len:
        return seq
    src_idx = np.linspace(0, T_in - 1, target_len)
    lo   = np.floor(src_idx).astype(int)
    hi   = np.minimum(lo + 1, T_in - 1)
    frac = (src_idx - lo).reshape(-1, 1, 1)
    return (seq[lo] * (1.0 - frac) + seq[hi] * frac).astype(np.float32)


def replicate_to_sequence(frame: np.ndarray, t: int = T) -> np.ndarray:
    """
    Replicate a single (21, 3) frame into a (T, 21, 3) pseudo-sequence.
    Augmentation noise applied per-epoch in the DataLoader creates variation
    across the replicated frames during training.
    """
    return np.tile(frame[np.newaxis], (t, 1, 1))  # (T, 21, 3)


def process_one(
    frame: np.ndarray,
    is_left: bool,
) -> np.ndarray:
    """Normalise → replicate. Returns (T, 21, 3) float32."""
    norm = normalize_frame(frame, is_left)
    return replicate_to_sequence(norm, T)


# ── Build and persist splits ──────────────────────────────────────────────────

def build_processed_dataset(
    train_sequences:  List[np.ndarray],
    train_feat_vecs:  List[np.ndarray],
    train_labels:     List[int],
    train_left_flags: List[bool],
    test_sequences:   List[np.ndarray],
    test_feat_vecs:   List[np.ndarray],
    test_labels:      List[int],
    test_left_flags:  List[bool],
    out_dir: Path = DATA_PROC,
) -> None:
    """
    Process all frames, split train into train+val, and save three .npz files.

    The Cabana dataset provides a pre-defined test split; only the train folder
    is divided into train / val (keeping VAL_SPLIT fraction for validation).

    Saved keys per .npz
    -------------------
    X       : (N, T, 21, 3) float32
    X_feat  : (N, 55)       float32
    y       : (N,)          int64
    """
    def _process_split(seqs, feats, labels, left_flags, desc):
        X_list, F_list = [], []
        for seq, is_left in tqdm(zip(seqs, left_flags), total=len(seqs),
                                  desc=desc):
            frame = seq[0]   # (21, 3) — sequences are single-frame (1, 21, 3)
            X_list.append(process_one(frame, is_left))
        # Features are already computed from raw (pre-normalisation) landmarks;
        # they are relative quantities so they are scale-invariant.
        F_list = list(feats)
        X = np.stack(X_list, axis=0).astype(np.float32)   # (N, T, 21, 3)
        F = np.stack(F_list, axis=0).astype(np.float32)   # (N, 55)
        y = np.array(labels, dtype=np.int64)               # (N,)
        return X, F, y

    print("[preprocess] Processing training images ...")
    X_tr, F_tr, y_tr = _process_split(
        train_sequences, train_feat_vecs, train_labels, train_left_flags,
        desc="Normalising train",
    )

    print("[preprocess] Processing test images ...")
    X_te, F_te, y_te = _process_split(
        test_sequences, test_feat_vecs, test_labels, test_left_flags,
        desc="Normalising test",
    )

    # Stratified val split from training data
    idx = np.arange(len(y_tr))
    idx_train, idx_val = train_test_split(
        idx, test_size=VAL_SPLIT, stratify=y_tr, random_state=SEED,
    )

    splits = {
        "train": (X_tr[idx_train], F_tr[idx_train], y_tr[idx_train]),
        "val":   (X_tr[idx_val],   F_tr[idx_val],   y_tr[idx_val]),
        "test":  (X_te,            F_te,            y_te),
    }

    for name, (X, F, y) in splits.items():
        out_path = out_dir / f"{name}.npz"
        np.savez_compressed(out_path, X=X, X_feat=F, y=y)
        print(f"[preprocess] {name:5s}: {len(y):6,} samples → {out_path}")

    print(f"\n[preprocess] Feature layout stored in X_feat (shape N×55):\n"
          f"  [0 :15] finger_angles      (15 joint angles, radians)\n"
          f"  [15:20] tip_wrist_dists    (5 fingertip-to-wrist L2 distances)\n"
          f"  [20:30] tip_tip_dists      (10 pairwise fingertip distances)\n"
          f"  [30:34] mcp_spread         (4 adjacent MCP lateral gaps)\n"
          f"  [34:37] palm_normal        (3-D unit palm normal vector)\n"
          f"  [37:42] finger_curl        (5 curl ratios, 1=extended 0=closed)\n"
          f"  [42:46] thumb_opposition   (4 thumb-tip to fingertip distances)\n"
          f"  [46:51] tip_heights        (5 fingertip z relative to palm)\n"
          f"  [51:55] dip_spread         (4 adjacent DIP lateral gaps)\n")


def load_split(
    split: str,
    data_dir: Path = DATA_PROC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a saved split.

    Returns
    -------
    X      : (N, T, 21, 3) float32
    y      : (N,)          int64
    """
    path = data_dir / f"{split}.npz"
    data = np.load(path)
    return data["X"], data["y"]


def load_split_with_features(
    split: str,
    data_dir: Path = DATA_PROC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a saved split including the derived feature matrix.

    Returns
    -------
    X      : (N, T, 21, 3) float32
    X_feat : (N, 55)       float32
    y      : (N,)          int64
    """
    path = data_dir / f"{split}.npz"
    data = np.load(path)
    return data["X"], data["X_feat"], data["y"]
