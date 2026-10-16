"""Central configuration (stub). Values follow the hardware audit: RTX 3050 Laptop, 4 GB VRAM."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
LOG_FILE = ROOT / "logs" / "progress.log"

CROPS = ["Tomato", "Apple", "Grape", "Strawberry", "Peach", "Cherry", "Orange", "Pepper_bell"]

SEED = 42
IMG_SIZE = 224          # GPU < 8 GB VRAM tier
BATCH_SIZE = 16         # user-approved increase from the 4 GB tier's 8 (VRAM use is ~0.4 GiB at b8); reduce on verified OOM and log
MAX_EPOCHS = 8
NUM_WORKERS = 3         # user-approved (0 was the Windows default); images are served from the 256px cache
MIXED_PRECISION = True
MAX_IMAGES_PER_CONDITION = 1000
SPLIT = (0.70, 0.15, 0.15)

LR_HEAD = 3e-4
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 3

SEVERITY_THRESHOLDS = (0.10, 0.30)  # damage_fraction: <0.10 Mild, 0.10-0.30 Moderate, >0.30 Severe (heuristic)
