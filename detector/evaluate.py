"""Calibrate decision thresholds and evaluate image-level anomaly scores."""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from detector.config import (
    IMAGE_EXTENSIONS,
    OUTPUT_DIR,
    TEST_ROOT,
    THRESHOLDS_PATH,
    TRAIN_GOOD,
    decision_for_score,
)
from detector.score import score_image


def evaluate() -> None:
    """Score held-out good images and all test images, then save thresholds."""
    index_meta_path = Path(__file__).resolve().parents[1] / "results" / "index" / "metadata.json"
    if not index_meta_path.is_file():
        raise FileNotFoundError("Build the patch index first with python -m detector.index.")
    with index_meta_path.open(encoding="utf-8") as file:
        validation_names = json.load(file)["validation_images"]
    validation_paths = [TRAIN_GOOD / name for name in validation_names]
    if not validation_paths or any(not path.is_file() for path in validation_paths):
        raise FileNotFoundError("One or more held-out good images from index metadata are missing.")

    test_paths = []
    for folder in sorted(path for path in TEST_ROOT.iterdir() if path.is_dir()):
        test_paths.extend(
            (path, folder.name)
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not test_paths:
        raise FileNotFoundError(f"No test images found under {TEST_ROOT}.")

    started = time.perf_counter()
    validation_scores = [score_image(path)[0] for path in validation_paths]
    test_records = [
        (path, defect_type, score_image(path)[0])
        for path, defect_type in test_paths
    ]
    average_seconds = (time.perf_counter() - started) / (
        len(validation_scores) + len(test_records)
    )

    good_scores = np.asarray(validation_scores)
    low_threshold = float(np.percentile(good_scores, 99))
    median = float(np.median(good_scores))
    mad = float(np.median(np.abs(good_scores - median)))
    # Three robust standard deviations above the good-image median create a
    # conservative "defect" boundary; the 99th-percentile-to-boundary gap is uncertain.
    margin = max(3 * 1.4826 * mad, max(abs(low_threshold), 1.0) * 1e-6)
    high_threshold = max(low_threshold + margin, median + margin)

    labels = [int(kind != "good") for _, kind, _ in test_records]
    scores = [score for _, _, score in test_records]
    if len(set(labels)) < 2:
        raise ValueError("Image AUROC needs both good and defective test images.")
    image_auroc = roc_auc_score(labels, scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }
    with THRESHOLDS_PATH.open("w", encoding="utf-8") as file:
        json.dump(thresholds, file, indent=2)

    print(f"Test image AUROC: {image_auroc:.4f}")
    print("AUROC by defect type (one type versus all other test images):")
    defect_types = sorted({kind for _, kind, _ in test_records if kind != "good"})
    for defect_type in defect_types:
        type_labels = [int(kind == defect_type) for _, kind, _ in test_records]
        if len(set(type_labels)) < 2:
            print(f"  {defect_type}: unavailable (needs positive and negative images)")
        else:
            print(f"  {defect_type}: {roc_auc_score(type_labels, scores):.4f}")

    print(f"Low threshold (99th percentile of good validation scores): {low_threshold:.6f}")
    print(f"High threshold (robust upper boundary): {high_threshold:.6f}")
    print(f"Average scoring time: {average_seconds:.4f} seconds/image")
    print("Test image decisions:")
    for path, defect_type, score in test_records:
        decision = decision_for_score(score, low_threshold, high_threshold)
        print(f"  {path.name} [{defect_type}]: {decision}")
    print(f"Saved thresholds to {THRESHOLDS_PATH}")


if __name__ == "__main__":
    evaluate()
