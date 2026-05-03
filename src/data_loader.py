"""
Load ASL static images from the Cabana 2025 dataset, extract hand skeleton
landmarks via MediaPipe HandLandmarker (tasks API), and compute rich derived
geometric features.

Dataset structure expected:
    src/datasets/.../ASL Alphabet Dataset Cabana 2025/
    ├── Train/
    │   ├── A/  *.jpg
    │   ├── B/  *.jpg
    │   └── ...
    └── Test/
        ├── A/  *.jpg
        └── ...

Primary output per image
------------------------
landmarks : (1, 21, 3) float32  — single-frame skeleton in normalised image
    coords (replicated to (T, 21, 3) during preprocessing)

Derived features per image (55 total, stored as X_feat in .npz)
---------------------------------------------------------------
[0 :15]  finger_angles      — joint angle at each of 15 knuckles (rad)
[15:20]  tip_wrist_dists    — L2 distance from each fingertip to wrist
[20:30]  tip_tip_dists      — pairwise fingertip distances (C(5,2)=10)
[30:34]  mcp_spread         — lateral spread between adjacent MCPs
[34:37]  palm_normal        — unit normal of the palm plane
[37:42]  finger_curl        — straight-dist / chain-length ratio per finger
[42:46]  thumb_opposition   — thumb-tip to each other fingertip distance
[46:51]  tip_heights        — fingertip z relative to palm-centre z
[51:55]  dip_spread         — lateral spread between adjacent DIPs
"""

import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_MAP, NUM_DERIVED_FEATURES, ROOT

# ── Model file (auto-downloaded on first run) ─────────────────────────────────
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_MODEL_PATH = ROOT / "data" / "hand_landmarker.task"

_BaseOptions          = mp.tasks.BaseOptions
_HandLandmarker       = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_RunningMode          = mp.tasks.vision.RunningMode


def _ensure_model() -> Path:
    """Download hand_landmarker.task if not already present."""
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data_loader] Downloading HandLandmarker model to {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        print("[data_loader] Model download complete.")
    return _MODEL_PATH


# ── MediaPipe hand landmark topology ──────────────────────────────────────────
# (parent, joint, child) — one bending knuckle per triplet
FINGER_JOINTS: List[Tuple[int, int, int]] = [
    # Thumb  (CMC→MCP, MCP→IP, IP→TIP)
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # Index  (wrist→MCP, MCP→PIP, PIP→DIP)
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # Middle
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # Ring
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # Pinky
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

FINGERTIP_IDX  = [4, 8, 12, 16, 20]
MCP_IDX        = [1, 5,  9, 13, 17]   # CMC for thumb; MCP for others
DIP_IDX        = [3, 7, 11, 15, 19]

# Ordered landmark chains per finger (base → tip)
FINGER_CHAINS = [
    (1,  2,  3,  4),    # Thumb
    (5,  6,  7,  8),    # Index
    (9,  10, 11, 12),   # Middle
    (13, 14, 15, 16),   # Ring
    (17, 18, 19, 20),   # Pinky
]

assert NUM_DERIVED_FEATURES == 55, "Feature layout constant mismatch"


# ── Geometric helpers ─────────────────────────────────────────────────────────

def _joint_angle(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Angle in radians at vertex q for the p-q-r triplet."""
    v1 = p - q
    v2 = r - q
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def compute_derived_features(lm: np.ndarray) -> np.ndarray:
    """
    Compute 55 geometric features from 21 MediaPipe hand landmarks.

    Parameters
    ----------
    lm : (21, 3) float32 — landmarks in any coordinate frame
         (all features are relative so the frame need not be normalised first)

    Returns
    -------
    (55,) float32
    """
    # 1. Finger joint angles [0:15]
    angles = np.array(
        [_joint_angle(lm[p], lm[q], lm[r]) for p, q, r in FINGER_JOINTS],
        dtype=np.float32,
    )

    # 2. Fingertip-to-wrist L2 distances [15:20]
    wrist = lm[0]
    tip_wrist = np.array(
        [np.linalg.norm(lm[i] - wrist) for i in FINGERTIP_IDX],
        dtype=np.float32,
    )

    # 3. Pairwise fingertip distances — C(5,2)=10 pairs [20:30]
    tips = lm[FINGERTIP_IDX]  # (5, 3)
    tip_tip = np.array(
        [np.linalg.norm(tips[i] - tips[j])
         for i in range(5) for j in range(i + 1, 5)],
        dtype=np.float32,
    )

    # 4. Lateral spread between adjacent MCPs [30:34]
    mcps = lm[MCP_IDX]  # (5, 3)
    mcp_spread = np.array(
        [np.linalg.norm(mcps[i] - mcps[i + 1]) for i in range(4)],
        dtype=np.float32,
    )

    # 5. Unit normal of the palm plane (wrist, index-MCP, pinky-MCP) [34:37]
    v1 = lm[5] - lm[0]
    v2 = lm[17] - lm[0]
    normal = np.cross(v1, v2)
    palm_normal = (normal / (np.linalg.norm(normal) + 1e-8)).astype(np.float32)

    # 6. Finger curl ratio: tip-to-base straight dist / sum-of-segment lengths [37:42]
    curl = []
    for chain in FINGER_CHAINS:
        straight  = np.linalg.norm(lm[chain[-1]] - lm[chain[0]])
        chain_len = sum(np.linalg.norm(lm[chain[k + 1]] - lm[chain[k]])
                        for k in range(len(chain) - 1))
        curl.append(straight / (chain_len + 1e-8))
    finger_curl = np.array(curl, dtype=np.float32)

    # 7. Thumb-tip to each non-thumb fingertip (opposition) [42:46]
    thumb_tip = lm[4]
    thumb_opp = np.array(
        [np.linalg.norm(thumb_tip - lm[i]) for i in [8, 12, 16, 20]],
        dtype=np.float32,
    )

    # 8. Fingertip z-heights relative to palm-centre z [46:51]
    palm_z = lm[[0, 5, 9, 13, 17], 2].mean()
    tip_heights = np.array(
        [lm[i, 2] - palm_z for i in FINGERTIP_IDX],
        dtype=np.float32,
    )

    # 9. Lateral spread between adjacent DIP joints [51:55]
    dips = lm[DIP_IDX]  # (5, 3)
    dip_spread = np.array(
        [np.linalg.norm(dips[i] - dips[i + 1]) for i in range(4)],
        dtype=np.float32,
    )

    feat = np.concatenate([
        angles,        # 15
        tip_wrist,     #  5
        tip_tip,       # 10
        mcp_spread,    #  4
        palm_normal,   #  3
        finger_curl,   #  5
        thumb_opp,     #  4
        tip_heights,   #  5
        dip_spread,    #  4
    ])                 # = 55
    assert feat.shape == (NUM_DERIVED_FEATURES,), feat.shape
    return feat


# ── MediaPipe extraction ──────────────────────────────────────────────────────

def _make_landmarker_options(model_path: Path):
    return _HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(model_path)),
        running_mode=_RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )


def _landmarks_from_result(result) -> Optional[Tuple[np.ndarray, bool]]:
    """Extract (21,3) array and handedness from a HandLandmarkerResult."""
    if not result.hand_landmarks:
        return None
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
        dtype=np.float32,
    )  # (21, 3)
    is_left = False
    if result.handedness:
        label = result.handedness[0][0].category_name  # "Left" or "Right"
        is_left = (label == "Left")
    return coords, is_left


def extract_landmarks_from_image(
    image_path: Path,
    landmarker,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Run MediaPipe HandLandmarker on one image file.

    Returns
    -------
    landmarks : (21, 3) float32 in normalised image coordinates, or None
    is_left   : True if MediaPipe detected a left hand
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None, False

    def _run(img):
        rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res   = landmarker.detect(mp_im)
        return _landmarks_from_result(res)

    out = _run(img_bgr)
    if out is not None:
        return out

    # Retry with horizontal flip — helps when image is mirrored / back-facing
    out = _run(cv2.flip(img_bgr, 1))
    if out is not None:
        coords, is_left = out
        return coords, not is_left   # flip inverts lateral labelling

    return None, False


# ── Dataset scanning ──────────────────────────────────────────────────────────

def scan_image_dataset(
    split_dir: Path,
) -> List[Tuple[Path, str, int]]:
    """
    Walk a Train/ or Test/ root and return
    (image_path, label_char, label_idx) for every image in a known class folder.
    """
    entries: List[Tuple[Path, str, int]] = []
    for letter_dir in sorted(split_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        label_char = letter_dir.name.strip().upper()
        if label_char not in LABEL_MAP:
            continue
        label_idx = LABEL_MAP[label_char]
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in sorted(letter_dir.glob(ext)):
                entries.append((img_path, label_char, label_idx))
    return entries


# ── Full dataset builder ──────────────────────────────────────────────────────

def build_dataset_from_images(
    split_dir: Path,
    desc: str = "Extracting landmarks",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[bool]]:
    """
    Extract MediaPipe landmarks and derived features from all images in split_dir.

    Parameters
    ----------
    split_dir : Path to Train/ or Test/ folder
    desc      : tqdm label

    Returns
    -------
    sequences  : list of (1, 21, 3) float32 — one frame per image
    feat_vecs  : list of (55,) float32       — derived geometric features
    labels     : list of int                 — class indices
    left_flags : list of bool                — True → mirror x in normaliser
    """
    entries   = scan_image_dataset(split_dir)
    n_classes = len({e[2] for e in entries})
    print(f"[data_loader] {desc}: {len(entries):,} images | {n_classes} classes")

    model_path = _ensure_model()
    options    = _make_landmarker_options(model_path)

    sequences:  List[np.ndarray] = []
    feat_vecs:  List[np.ndarray] = []
    labels:     List[int]        = []
    left_flags: List[bool]       = []
    skipped = 0

    with _HandLandmarker.create_from_options(options) as landmarker:
        for img_path, _, label_idx in tqdm(entries, desc=desc):
            out = extract_landmarks_from_image(img_path, landmarker)
            if out is None or out[0] is None:
                skipped += 1
                continue
            lm, is_left = out

            sequences.append(lm[np.newaxis].copy())         # (1, 21, 3)
            feat_vecs.append(compute_derived_features(lm))  # (55,)
            labels.append(label_idx)
            left_flags.append(is_left)

    n_ok = len(sequences)
    pct  = 100.0 * skipped / max(len(entries), 1)
    print(f"[data_loader] Done — {n_ok:,} extracted, "
          f"{skipped:,} skipped ({pct:.1f}% detection failures).")
    return sequences, feat_vecs, labels, left_flags
