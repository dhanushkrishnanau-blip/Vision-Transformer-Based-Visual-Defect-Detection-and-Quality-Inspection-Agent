"""Score images and save visual overlays of their anomaly maps."""

import json
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from scipy.ndimage import gaussian_filter

from detector.config import IMAGE_SIZE, INDEX_DIR
from detector.features import extract_patch_features

_index = None
_metadata = None


def _load_index():
    """Load the saved FAISS index and its settings once."""
    global _index, _metadata
    if _index is None:
        index_path = INDEX_DIR / "patches.faiss"
        metadata_path = INDEX_DIR / "metadata.json"
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Build the patch index first with python -m detector.index.")
        _index = faiss.read_index(str(index_path))
        with metadata_path.open(encoding="utf-8") as file:
            _metadata = json.load(file)
        if _metadata["image_size"] != IMAGE_SIZE:
            raise ValueError("Index image size does not match the current detector settings.")
    return _index


def score_image(path: str | Path) -> tuple[float, np.ndarray]:
    """Return the maximum patch distance and a blurred map at original image size."""
    index = _load_index()
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size

    features = extract_patch_features([image]).cpu().numpy()[0].astype("float32")
    distances, _ = index.search(features, 1)
    patch_distances = distances[:, 0]
    grid_size = int(np.sqrt(len(patch_distances)))
    if grid_size * grid_size != len(patch_distances):
        raise ValueError(f"Patch count {len(patch_distances)} cannot form a square map.")

    # Resize the patch grid to the source image, then smooth blocky boundaries.
    grid = torch.from_numpy(patch_distances).reshape(1, 1, grid_size, grid_size)
    resized = functional.interpolate(
        grid, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    anomaly_map = gaussian_filter(resized, sigma=max(1, IMAGE_SIZE / 56)).astype("float32")
    return float(patch_distances.max()), anomaly_map


def save_heatmap(path: str | Path, anomaly_map: np.ndarray, out_path: str | Path) -> None:
    """Save a color heatmap blended over the original image."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    values = np.asarray(anomaly_map, dtype="float32")
    spread = float(np.ptp(values))
    normalized = (values - float(values.min())) / spread if spread > 0 else np.zeros_like(values)
    colored = cv2.applyColorMap((normalized * 255).astype("uint8"), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = Image.blend(image, Image.fromarray(colored), alpha=0.4)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output)
