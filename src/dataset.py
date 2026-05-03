"""
PyTorch Dataset wrapping the processed .npz splits.

Returns tensors:
    x : (T, 21, 3) float32  — normalised keypoint sequence
    y : ()          int64   — class index
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import T, NUM_KEYPOINTS, COORDS, BATCH_SIZE, SEED
from src.preprocess   import load_split
from src.augmentation import SkeletonAugmenter


class ASLDataset(Dataset):
    def __init__(self,
                 split:     str,
                 augment:   bool = False,
                 train_X:   Optional[np.ndarray] = None,
                 train_y:   Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        split    : 'train' | 'val' | 'test'
        augment  : apply SkeletonAugmenter (train split only)
        train_X  : pre-loaded training sequences (needed to build MixSkel pool)
        train_y  : pre-loaded training labels
        """
        self.X, self.y = load_split(split)
        self.augment   = augment

        # Build class pool from training data for MixSkel
        class_pool: dict = defaultdict(list)
        if augment and train_X is not None and train_y is not None:
            for seq, lbl in zip(train_X, train_y):
                class_pool[int(lbl)].append(seq)

        self.augmenter = SkeletonAugmenter(class_pool=class_pool,
                                            training=augment)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        seq = self.X[idx]                          # (T, 21, 3)
        lbl = int(self.y[idx])

        if self.augment:
            seq = self.augmenter(seq, lbl)

        x = torch.from_numpy(seq)                  # (T, 21, 3)
        y = torch.tensor(lbl, dtype=torch.long)
        return x, y


def get_dataloaders(num_workers: int = 0):
    """
    Build and return train / val / test DataLoaders.
    """
    # Load training data first so augmenter can build the MixSkel pool
    train_X, train_y = load_split("train")

    train_ds = ASLDataset("train", augment=True,
                           train_X=train_X, train_y=train_y)
    val_ds   = ASLDataset("val",   augment=False)
    test_ds  = ASLDataset("test",  augment=False)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=num_workers,
                               pin_memory=torch.cuda.is_available(), generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=torch.cuda.is_available())

    print(f"[dataset] train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}")
    return train_loader, val_loader, test_loader
