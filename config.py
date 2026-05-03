from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_RAW    = ROOT / "data" / "raw"          # legacy (unused with Cabana dataset)
DATA_PROC   = ROOT / "data" / "processed"    # normalised .npz files
CHECKPOINTS = ROOT / "checkpoints"
LOGS        = ROOT / "logs"

# Cabana 2025 image dataset
_CABANA_BASE     = (ROOT / "src" / "datasets"
                    / "American Sign Language Alphabet Dataset"
                    / "American Sign Language Alphabet Dataset"
                    / "ASL Alphabet Dataset Cabana 2025")
CABANA_TRAIN_PATH = _CABANA_BASE / "Train"
CABANA_TEST_PATH  = _CABANA_BASE / "Test"

for _p in [DATA_RAW, DATA_PROC, CHECKPOINTS, LOGS]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Sequence ───────────────────────────────────────────────────────────────────
T             = 30        # pseudo-sequence length (single frame replicated T times)
NUM_KEYPOINTS = 21        # MediaPipe hand landmarks
COORDS        = 3         # x, y, z
FEATURE_DIM   = NUM_KEYPOINTS * COORDS   # 63  (raw landmark flat dim)

# Derived geometric features computed per image (stored in X_feat in .npz files)
# Layout: [finger_angles(15) | tip_wrist_dists(5) | tip_tip_dists(10) |
#          mcp_spread(4) | palm_normal(3) | finger_curl(5) |
#          thumb_opposition(4) | tip_heights(5) | dip_spread(4)]
NUM_DERIVED_FEATURES = 55

# ── Classes ────────────────────────────────────────────────────────────────────
# 24 static ASL letters (J and Z require motion — excluded)
STATIC_LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]   # 24 letters
NUM_CLASSES    = len(STATIC_LETTERS)                        # 24
LABEL_MAP      = {ch: i for i, ch in enumerate(STATIC_LETTERS)}
IDX_TO_LABEL   = {i: ch for ch, i in LABEL_MAP.items()}

# ── Hand skeleton graph (MediaPipe 21 landmarks) ───────────────────────────────
# Each tuple is a directed edge (src, dst); we add both directions in models.
HAND_EDGES = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),       # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),       # index
    (0, 9),  (9, 10), (10,11), (11,12),      # middle
    (0,13), (13,14), (14,15), (15,16),       # ring
    (0,17), (17,18), (18,19), (19,20),       # pinky
    (5, 9),  (9,13), (13,17),               # palm cross-connections
]

# ── Confusable clusters (used for hard-pair augmentation & evaluation) ─────────
CONFUSABLE_CLUSTERS = [
    ["A", "S", "E", "T", "N", "M"],    # closed-fist family
    ["K", "R", "U", "V", "W"],         # extended-finger family
]
HARD_CLASSES = {ch for cluster in CONFUSABLE_CLUSTERS for ch in cluster}

# ── Augmentation ───────────────────────────────────────────────────────────────
AUG_ROTATION_DEG  = 15      # ±15° in-plane rotation
AUG_SCALE_RANGE   = (0.85, 1.15)
AUG_SPEED_RANGE   = (0.80, 1.20)
AUG_NOISE_SIGMA   = 0.02
AUG_MIXSKEL_ALPHA = (0.30, 0.70)
AUG_PROB_NORMAL   = 0.50    # probability for optional ops on normal classes
AUG_PROB_HARD     = 0.80    # probability for optional ops on hard classes

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
NUM_EPOCHS    = 100
LR            = 1e-3
LR_MIN        = 1e-5        # cosine annealing floor
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15          # early-stopping patience
LABEL_SMOOTH  = 0.10
SUPCON_LAMBDA = 0.30        # weight of contrastive loss (M4, M5 only)
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.10
SEED          = 42

# ── Model hidden dims ──────────────────────────────────────────────────────────
BILSTM_HIDDEN  = 256
TCN_CHANNELS   = 128
TRANS_D_MODEL  = 128
TRANS_HEADS    = 4
TRANS_LAYERS   = 4
TRANS_FF_DIM   = 256
STGCN_CHANNELS = [64, 128, 256]
GAT_HEADS      = 4
GAT_D_MODEL    = 128
