"""
Evaluation: load best checkpoint → run test set → report metrics.

Outputs
-------
- Overall accuracy
- Per-class accuracy table
- Confusion matrix (saved as PNG)
- Confusable-cluster accuracy breakdown

Usage
-----
python src/evaluate.py --model gat
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (NUM_CLASSES, CHECKPOINTS, LOGS,
                    IDX_TO_LABEL, CONFUSABLE_CLUSTERS, LABEL_MAP)
from src.dataset import get_dataloaders
from src.models  import MODEL_REGISTRY


def load_best(model_name: str, device: torch.device):
    ckpt_path = CHECKPOINTS / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = MODEL_REGISTRY[model_name]().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[evaluate] Loaded epoch={ckpt['epoch']}  "
          f"val_acc={ckpt['val_acc']:.4f}")
    return model


@torch.no_grad()
def predict(model, loader, device):
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits, _ = model(x)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                           save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix → {save_path}")


def cluster_accuracy(preds: np.ndarray, labels: np.ndarray) -> None:
    """Print per-class and per-cluster accuracy for confusable clusters."""
    print("\n── Confusable Cluster Breakdown ──────────────────────────────")
    for cluster in CONFUSABLE_CLUSTERS:
        cluster_idx = [LABEL_MAP[c] for c in cluster if c in LABEL_MAP]
        mask        = np.isin(labels, cluster_idx)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        print(f"  {cluster}  →  cluster acc = {acc:.4f}  "
              f"(n={mask.sum()})")
        for idx in cluster_idx:
            m2 = labels == idx
            if m2.sum() == 0:
                continue
            a2 = (preds[m2] == idx).mean()
            print(f"      {IDX_TO_LABEL[idx]}  per-class acc = {a2:.4f}  "
                  f"(n={m2.sum()})")


def run_evaluation(model_name: str = "gat", device_str: str = "auto"):
    device = (torch.device("cuda")
              if device_str == "auto" and torch.cuda.is_available()
              else torch.device(device_str if device_str != "auto" else "cpu"))

    model = load_best(model_name, device)
    _, _, test_loader = get_dataloaders()
    preds, labels     = predict(model, test_loader, device)

    # ── Overall ──────────────────────────────────────────────────────────────
    overall_acc = (preds == labels).mean()
    print(f"\n[evaluate] Overall test accuracy: {overall_acc:.4f}")

    # ── Per-class report ─────────────────────────────────────────────────────
    class_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    print("\n── Classification Report ─────────────────────────────────────")
    print(classification_report(labels, preds, target_names=class_names,
                                 digits=4))

    # ── Confusion matrix ───────────────────────────────────────────────���─────
    cm       = confusion_matrix(labels, preds)
    cm_path  = LOGS / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    # ── Cluster breakdown ────────────────────────────────────────────────────
    cluster_accuracy(preds, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="gat",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_evaluation(args.model, args.device)
