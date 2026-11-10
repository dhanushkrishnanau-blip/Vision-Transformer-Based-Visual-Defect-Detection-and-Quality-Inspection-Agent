"""Shared paths and settings for the DINOv2 detector."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec" / "bottle"
TRAIN_GOOD = DATA_ROOT / "train" / "good"
TEST_ROOT = DATA_ROOT / "test"
INDEX_DIR = PROJECT_ROOT / "results" / "index"
OUTPUT_DIR = PROJECT_ROOT / "results" / "output"
THRESHOLDS_PATH = OUTPUT_DIR / "thresholds.json"

IMAGE_SIZE = 224
MODEL_NAME = "facebook/dinov2-small"
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_thresholds() -> dict[str, float] | None:
    """Read evaluation thresholds when they have already been saved."""
    if not THRESHOLDS_PATH.exists():
        return None
    with THRESHOLDS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def decision_for_score(score: float, low: float, high: float) -> str:
    """Map a score to normal, uncertain, or defect."""
    if score <= low:
        return "normal"
    if score <= high:
        return "uncertain"
    return "defect"


THRESHOLDS = load_thresholds()
