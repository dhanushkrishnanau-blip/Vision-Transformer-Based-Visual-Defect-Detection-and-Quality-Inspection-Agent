"""Calibrate decision thresholds and evaluate image-level anomaly scores."""

import json
import time
from collections import Counter, defaultdict
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
    print("AUROC by defect type (that defect type versus good test images):")
    good_records = [record for record in test_records if record[1] == "good"]
    defect_types = sorted({kind for _, kind, _ in test_records if kind != "good"})
    for defect_type in defect_types:
        comparison_records = good_records + [
            record for record in test_records if record[1] == defect_type
        ]
        type_labels = [int(kind == defect_type) for _, kind, _ in comparison_records]
        type_scores = [score for _, _, score in comparison_records]
        if not good_records or len(set(type_labels)) < 2:
            print(f"  {defect_type}: unavailable (needs positive and negative images)")
        else:
            print(f"  {defect_type}: {roc_auc_score(type_labels, type_scores):.4f}")

    print(f"Low threshold (99th percentile of good validation scores): {low_threshold:.6f}")
    print(f"High threshold (robust upper boundary): {high_threshold:.6f}")
    print(f"Average scoring time: {average_seconds:.4f} seconds/image")

    counts_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    missed_defects = 0
    false_alarms = 0
    for _, defect_type, score in test_records:
        decision = decision_for_score(score, low_threshold, high_threshold)
        counts_by_type[defect_type][decision] += 1
        if defect_type != "good" and decision != "defect":
            missed_defects += 1
        elif defect_type == "good" and decision == "defect":
            false_alarms += 1

    print("Test image decisions by type:")
    print(f"  {'Type':<20} {'Normal':>8} {'Uncertain':>10} {'Defect':>8} {'Total':>8}")
    for image_type in sorted(counts_by_type):
        counts = counts_by_type[image_type]
        total = sum(counts.values())
        print(
            f"  {image_type:<20} {counts['normal']:>8} "
            f"{counts['uncertain']:>10} {counts['defect']:>8} {total:>8}"
        )
    print(f"Missed defects (not labelled defect): {missed_defects}")
    print(f"False alarms (good images labelled defect): {false_alarms}")

    print("Test image decisions:")
    for path, defect_type, score in test_records:
        decision = decision_for_score(score, low_threshold, high_threshold)
        print(f"  {path.name} [{defect_type}]: {decision}")
    print(f"Saved thresholds to {THRESHOLDS_PATH}")


if __name__ == "__main__":
    evaluate()
