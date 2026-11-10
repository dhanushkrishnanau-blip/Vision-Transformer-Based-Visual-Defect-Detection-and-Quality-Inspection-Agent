"""Build a FAISS nearest-neighbor index from normal training images."""

import argparse
import json
import random

import faiss
import numpy as np
from PIL import Image

from detector.config import (
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    INDEX_DIR,
    MODEL_NAME,
    RANDOM_SEED,
    TRAIN_GOOD,
)
from detector.features import extract_patch_features


def build_index(max_patches: int | None = None) -> None:
    """Hold out 10% of good images, then index patches from the remaining 90%."""
    if max_patches is not None and max_patches < 1:
        raise ValueError("max_patches must be a positive integer.")

    images = sorted(
        path for path in TRAIN_GOOD.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) < 2:
        raise FileNotFoundError(f"Need at least two training images in {TRAIN_GOOD}.")

    # Shuffle with a fixed seed so the same images are held out every run.
    random.Random(RANDOM_SEED).shuffle(images)
    validation_count = min(len(images) - 1, max(1, round(len(images) * 0.1)))
    validation_images = images[:validation_count]
    training_images = images[validation_count:]

    feature_batches = []
    for start in range(0, len(training_images), 8):
        batch_paths = training_images[start : start + 8]
        batch = []
        for path in batch_paths:
            with Image.open(path) as image:
                batch.append(image.convert("RGB"))
        features = extract_patch_features(batch).cpu().numpy()
        feature_batches.append(features.reshape(-1, features.shape[-1]))

    patches = np.concatenate(feature_batches).astype("float32")
    if max_patches is not None and len(patches) > max_patches:
        # Sample patches rather than images to keep large indexes manageable.
        rng = np.random.default_rng(RANDOM_SEED)
        patches = patches[rng.choice(len(patches), max_patches, replace=False)]

    index = faiss.IndexFlatL2(patches.shape[1])
    index.add(patches)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "patches.faiss"))

    metadata = {
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "seed": RANDOM_SEED,
        "indexed_images": len(training_images),
        "indexed_patches": len(patches),
        "max_patches": max_patches,
        "validation_images": [path.name for path in validation_images],
    }
    with (INDEX_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Indexed {len(patches):,} patches from {len(training_images)} images.")
    print(f"Held out {len(validation_images)} good images for threshold calibration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-patches", type=int, default=None)
    build_index(parser.parse_args().max_patches)
