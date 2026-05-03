"""
Training loop with:
  - cosine LR annealing
  - early stopping on validation loss
  - model checkpointing
  - CombinedLoss (CE + optional SupCon for M4/M5)
  - per-epoch logging to CSV

Usage
-----
python src/train.py --model gat --epochs 100
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (NUM_EPOCHS, LR, LR_MIN, WEIGHT_DECAY,
                    PATIENCE, CHECKPOINTS, LOGS, SEED, NUM_CLASSES)
from src.dataset import get_dataloaders
from src.losses  import CombinedLoss
from src.models  import MODEL_REGISTRY


SUPCON_MODELS = {"stgcn", "gat"}   # these receive contrastive loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, emb = model(x)
        loss = criterion(logits, y, emb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, emb = model(x)
        loss = criterion(logits, y, emb)

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


def train(model_name: str = "gat",
          epochs:     int = NUM_EPOCHS,
          device_str: str = "auto"):

    torch.manual_seed(SEED)
    device = (torch.device("cuda") if device_str == "auto" and torch.cuda.is_available()
              else torch.device(device_str if device_str != "auto" else "cpu"))
    print(f"[train] device={device}  model={model_name}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    ModelClass = MODEL_REGISTRY[model_name]
    model      = ModelClass().to(device)
    print(f"[train] Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Loss, Optimiser, Scheduler ────────────────────────────────────────────
    use_supcon = model_name in SUPCON_MODELS
    criterion  = CombinedLoss(use_supcon=use_supcon)
    optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR_MIN)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path = LOGS / f"{model_name}_log.csv"
    ckpt_path = CHECKPOINTS / f"{model_name}_best.pt"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr", "elapsed_s"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                           criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f} | "
              f"lr {lr_now:.2e} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_acc,
                                     va_loss, va_acc, lr_now, elapsed])

        # Checkpoint
        if va_loss < best_val_loss:
            best_val_loss  = va_loss
            patience_count = 0
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_loss":    va_loss,
                        "val_acc":     va_acc}, ckpt_path)
            print(f"           ✓ checkpoint saved (val_loss={va_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"[train] Early stopping at epoch {epoch}.")
                break

    print(f"[train] Best val_loss={best_val_loss:.4f} | "
          f"checkpoint → {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="gat",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(args.model, args.epochs, args.device)
