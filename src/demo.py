"""
Real-time ASL static sign recognition — webcam demo.

Loads a trained model checkpoint and classifies the ASL hand sign
shown to the webcam, overlaying the skeleton and a prediction panel.

Usage
-----
python src/demo.py --model gat
python src/demo.py --model bilstm --camera 1
python src/demo.py --model gat --device cpu
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHECKPOINTS, IDX_TO_LABEL, NUM_CLASSES
from src.data_loader import (_ensure_model, _make_landmarker_options,
                              _HandLandmarker, _landmarks_from_result)
from src.preprocess  import normalize_frame, replicate_to_sequence
from src.models      import MODEL_REGISTRY


# Hand skeleton edges for drawing (MediaPipe 21-landmark topology)
_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),        # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),        # index
    (0, 9),  (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9),  (9, 13),  (13, 17),              # palm cross-bar
]


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model_name: str, device: torch.device):
    ckpt_path = CHECKPOINTS / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first with:  python src/pipeline.py --model {model_name}"
        )
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = MODEL_REGISTRY[model_name]().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[demo] Loaded {model_name!r}  |  "
          f"epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.3f}  "
          f"device={device}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer(model, lm_raw: np.ndarray, is_left: bool,
           device: torch.device) -> tuple:
    """
    Run one forward pass.

    Returns
    -------
    label : str   — predicted ASL letter
    probs : (N,)  — softmax probabilities for all classes
    """
    frame = normalize_frame(lm_raw, is_left)       # (21, 3) normalised
    seq   = replicate_to_sequence(frame)            # (T, 21, 3)
    x     = torch.from_numpy(seq).unsqueeze(0).to(device)   # (1, T, 21, 3)
    logits, _ = model(x)
    probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (NUM_CLASSES,)
    label  = IDX_TO_LABEL[int(probs.argmax())]
    return label, probs


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_skeleton(frame: np.ndarray,
                   lm_raw: np.ndarray,
                   img_h: int, img_w: int) -> None:
    """Draw hand skeleton lines and landmark dots (in-place)."""
    # lm_raw: (21, 3) with x,y in [0,1] normalised image coords
    pts = [(int(lm[0] * img_w), int(lm[1] * img_h)) for lm in lm_raw]

    for i, j in _CONNECTIONS:
        cv2.line(frame, pts[i], pts[j], (0, 220, 0), 2, cv2.LINE_AA)

    for k, pt in enumerate(pts):
        colour = (0, 60, 255) if k == 0 else (255, 255, 255)  # wrist = red
        cv2.circle(frame, pt, 4, colour, -1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray,
              label: str,
              probs: np.ndarray,
              fps: float,
              top_n: int = 5,
              hand_visible: bool = True) -> None:
    """Overlay prediction panel on the right side of the frame (in-place)."""
    h, w = frame.shape[:2]
    pw   = 240   # panel width

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - pw, 0), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if hand_visible:
        # Big predicted letter
        cv2.putText(frame, label,
                    (w - pw + 50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.5,
                    (45, 255, 110), 9, cv2.LINE_AA)

        # Top-N probability bars
        top_idx = probs.argsort()[::-1][:top_n]
        for rank, idx in enumerate(top_idx):
            lbl_str  = IDX_TO_LABEL[int(idx)]
            prob_val = float(probs[idx])
            y0       = 165 + rank * 50
            bar_len  = int(prob_val * (pw - 55))
            bar_col  = (35, 210, 75) if rank == 0 else (70, 70, 70)

            cv2.rectangle(frame,
                          (w - pw + 10, y0),
                          (w - pw + 10 + bar_len, y0 + 30),
                          bar_col, -1)
            cv2.putText(frame,
                        f"{lbl_str}   {prob_val * 100:4.1f}%",
                        (w - pw + 14, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (240, 240, 240), 1, cv2.LINE_AA)
    else:
        # No hand detected — show placeholder
        cv2.putText(frame, "No hand",
                    (w - pw + 20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (100, 100, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, "detected",
                    (w - pw + 20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (100, 100, 100), 2, cv2.LINE_AA)

    # FPS counter (bottom-left)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (160, 160, 160), 1, cv2.LINE_AA)

    # Quit hint
    cv2.putText(frame, "Q = quit",
                (10, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 120, 120), 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_demo(model_name: str = "gat",
             camera_idx: int  = 0,
             device_str: str  = "auto",
             smooth_n:   int  = 5) -> None:
    """
    Parameters
    ----------
    model_name : one of MODEL_REGISTRY keys (gat, bilstm, tcn, transformer, stgcn)
    camera_idx : OpenCV camera index (0 = built-in webcam)
    device_str : 'auto' | 'cuda' | 'cpu'
    smooth_n   : number of recent predictions to average for stable output
    """
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Model
    model = _load_model(model_name, device)

    # MediaPipe landmarker (kept open for the whole session)
    mp_model_path = _ensure_model()
    mp_opts       = _make_landmarker_options(mp_model_path)
    landmarker    = _HandLandmarker.create_from_options(mp_opts)

    # Webcam
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(f"Cannot open camera index {camera_idx}. "
                           f"Try --camera 1 if you have multiple cameras.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"[demo] Camera {camera_idx} opened. Press Q to quit.")

    # Smoothing buffer: average probabilities over last smooth_n frames
    prob_buf   = deque(maxlen=smooth_n)
    label      = "—"
    probs      = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
    hand_vis   = False
    prev_t     = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[demo] Frame capture failed — exiting.")
                break

            img_h, img_w = frame.shape[:2]

            # Run MediaPipe on the current frame
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)
            det    = _landmarks_from_result(result)

            if det is not None:
                lm_raw, is_left = det
                hand_vis = True
                _draw_skeleton(frame, lm_raw, img_h, img_w)

                # Inference + smoothing
                _, frame_probs = _infer(model, lm_raw, is_left, device)
                prob_buf.append(frame_probs)
                probs = np.mean(prob_buf, axis=0)
                label = IDX_TO_LABEL[int(probs.argmax())]
            else:
                hand_vis = False
                prob_buf.clear()

            # FPS
            now    = time.perf_counter()
            fps    = 1.0 / max(now - prev_t, 1e-9)
            prev_t = now

            _draw_hud(frame, label, probs, fps, hand_visible=hand_vis)
            cv2.imshow(f"ASL Demo  [{model_name.upper()}]", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("[demo] Closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time ASL webcam demo")
    parser.add_argument("--model",  default="gat",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Trained model to load (default: gat)")
    parser.add_argument("--camera", type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    parser.add_argument("--device", default="auto",
                        help="'auto' | 'cuda' | 'cpu' (default: auto)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Frames to average for stable prediction (default: 5)")
    args = parser.parse_args()
    run_demo(args.model, args.camera, args.device, args.smooth)
