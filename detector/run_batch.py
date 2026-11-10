"""Score a folder of images and write a CSV for the future inspection agent."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from detector.config import (
    IMAGE_EXTENSIONS,
    OUTPUT_DIR,
    decision_for_score,
    load_thresholds,
)
from detector.score import save_heatmap, score_image


def run_batch(image_folder: Path) -> None:
    """Save one heatmap and CSV row for every image under a folder."""
    if not image_folder.is_dir():
        raise NotADirectoryError(f"Image folder does not exist: {image_folder}")
    thresholds = load_thresholds()
    if thresholds is None:
        raise FileNotFoundError("Run python -m detector.evaluate to create thresholds first.")
    low = float(thresholds["low_threshold"])
    high = float(thresholds["high_threshold"])
    heatmap_root = OUTPUT_DIR / "heatmaps"
    csv_path = OUTPUT_DIR / "detections.csv"
    images = sorted(
        path for path in image_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {image_folder}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "heatmap_path", "score", "decision", "timestamp"],
        )
        writer.writeheader()
        for image_path in images:
            score, anomaly_map = score_image(image_path)
            heatmap_path = heatmap_root / image_path.relative_to(image_folder).with_suffix(".png")
            save_heatmap(image_path, anomaly_map, heatmap_path)
            writer.writerow(
                {
                    "image_path": str(image_path.resolve()),
                    "heatmap_path": str(heatmap_path.resolve()),
                    "score": f"{score:.6f}",
                    "decision": decision_for_score(score, low, high),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
    print(f"Scored {len(images)} images. Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_folder", type=Path)
    run_batch(parser.parse_args().image_folder.resolve())
