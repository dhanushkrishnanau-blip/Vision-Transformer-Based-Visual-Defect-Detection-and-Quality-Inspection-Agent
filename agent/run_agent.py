"""Run detections from demo_data/detections.csv through the inspection agent."""

import argparse
import csv
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from agent.db import InMemoryDB, SupabaseDB
from agent.graph import run_detection


def _clean_path(value: str, marker: str) -> str:
    """Keep only the relative path after a known Colab folder marker."""
    parts = [part for part in value.replace("\\", "/").split("/") if part not in {"", "."}]
    marker_parts = marker.split("/")
    for index in range(len(parts) - len(marker_parts) + 1):
        if [part.lower() for part in parts[index : index + len(marker_parts)]] == [
            part.lower() for part in marker_parts
        ]:
            parts = parts[index + len(marker_parts) :]
            break
    safe_parts = [part for part in parts if part != ".."]
    return str(PurePosixPath(*safe_parts))


def load_detections(limit: int | None = None) -> list[dict[str, str]]:
    """Read CSV rows and normalize paths before the database sees them."""
    detections = []
    with open("demo_data/detections.csv", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "heatmap_path", "score", "decision"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("detections.csv is missing one or more required columns.")
        for row in reader:
            detections.append(
                {
                    "image_path": _clean_path(row["image_path"], "data/mvtec"),
                    "heatmap_path": _clean_path(row["heatmap_path"], "heatmaps"),
                    "score": row["score"],
                    # The routing result comes only from this detector-provided column.
                    "decision": row["decision"].strip().lower(),
                    "timestamp": row.get("timestamp", ""),
                }
            )
            if limit is not None and len(detections) >= limit:
                break
    return detections


def run(offline: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Process the CSV and print a count-only summary."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")
    db = InMemoryDB() if offline else SupabaseDB()
    detections = load_detections(limit)
    totals: Counter[str] = Counter()
    errors = 0

    for detection in detections:
        result = run_detection(db, detection)
        totals[detection["decision"]] += 1
        totals["tickets_created"] += int(result.get("ticket_created", False))
        totals["reviews_created"] += int(result.get("review_created", False))
        if result.get("error"):
            errors += 1

    print(f"Processed detections: {len(detections)}")
    print(f"Detector decisions: normal={totals['normal']}, "
          f"uncertain={totals['uncertain']}, defect={totals['defect']}")
    print(f"New tickets: {totals['tickets_created']}")
    print(f"New reviews: {totals['reviews_created']}")
    print(f"Cases with tool errors: {errors}")
    return {"processed": len(detections), "totals": totals, "errors": errors, "db": db}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use memory only; do not contact Supabase.")
    parser.add_argument("--limit", type=int, help="Process only the first N CSV rows.")
    args = parser.parse_args()
    result = run(offline=args.offline, limit=args.limit)

    if not args.offline:
        print("Supabase row counts:")
        for table in ("inspections", "tickets", "reviews", "log"):
            print(f"  {table}: {result['db'].count(table)}")


if __name__ == "__main__":
    main()
