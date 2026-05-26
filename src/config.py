from pathlib import Path
import torch

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "data" / "dataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
EXPORT_DIR = BASE_DIR / "exports"

CHECKPOINT_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# ---------------- TRAINING ----------------
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# TEMP DEV SETTING
EPOCHS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- IMAGE SIZES ----------------
RESNET_IMAGE_SIZE = 128
FUSION_IMAGE_SIZE = 260
YOLO_IMAGE_SIZE = 640

# ---------------- YOLO ----------------
YOLO_BASE_MODEL = "yolo11m.pt"
YOLO_BATCH_SIZE = 10
YOLO_EPOCHS = 1
YOLO_CONFIDENCE_THRESHOLD = 0.05

# ---------------- CLASSES ----------------
CLASS_NAMES = [
    "F_Breakage",
    "F_Crushed",
    "F_Normal",
    "R_Breakage",
    "R_Crushed",
    "R_Normal"
]



CLASS_MAP = {idx: cls for idx, cls in enumerate(CLASS_NAMES)}
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

NUM_CLASSES = len(CLASS_NAMES)

# ---------------- HUGGING FACE ----------------
HF_USERNAME = "junaid17"

HF_RESNET_REPO = "new-car-damage-classifier"
HF_FUSION_REPO = "new-best-fusion-model-fp16"
HF_YOLO_REPO = "new-Yolo-Model"