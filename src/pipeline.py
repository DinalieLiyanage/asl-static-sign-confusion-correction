"""
End-to-end pipeline runner.

Steps
-----
1. Extract MediaPipe landmarks + derived features from Train/ images
2. Extract MediaPipe landmarks + derived features from Test/ images
3. Normalise, replicate frames to T=30, split train→train+val, save .npz
4. Train model
5. Evaluate model on test set

Usage
-----
python src/pipeline.py --model gat
python src/pipeline.py --model bilstm --epochs 50
python src/pipeline.py --model gat --skip-preprocess   # reuse cached .npz
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROC, CABANA_TRAIN_PATH, CABANA_TEST_PATH
from src.data_loader import build_dataset_from_images
from src.preprocess  import build_processed_dataset
from src.train       import train
from src.evaluate    import run_evaluation
from src.models      import MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="ASL static sign recognition pipeline")
    parser.add_argument("--model",  default="gat",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip MediaPipe extraction if .npz files already exist")
    args = parser.parse_args()

    # ── Step 1-3: Feature extraction & preprocessing ──────────────────────────
    if args.skip_preprocess and (DATA_PROC / "train.npz").exists():
        print("[pipeline] Processed .npz files found — skipping extraction.")
    else:
        print("\n── Step 1: Extract landmarks from Train/ images ─────────────")
        tr_seqs, tr_feats, tr_labels, tr_left = build_dataset_from_images(
            CABANA_TRAIN_PATH, desc="Train extraction",
        )

        print("\n── Step 2: Extract landmarks from Test/ images ──────────────")
        te_seqs, te_feats, te_labels, te_left = build_dataset_from_images(
            CABANA_TEST_PATH, desc="Test extraction",
        )

        print("\n── Step 3: Normalise, replicate frames, save splits ─────────")
        build_processed_dataset(
            train_sequences=tr_seqs,
            train_feat_vecs=tr_feats,
            train_labels=tr_labels,
            train_left_flags=tr_left,
            test_sequences=te_seqs,
            test_feat_vecs=te_feats,
            test_labels=te_labels,
            test_left_flags=te_left,
        )

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    print(f"\n── Step 4: Train  model={args.model} ────────────────────────")
    train(args.model, args.epochs, args.device)

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print(f"\n── Step 5: Evaluate  model={args.model} ─────────────────────")
    run_evaluation(args.model, args.device)


if __name__ == "__main__":
    main()
