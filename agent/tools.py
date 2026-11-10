"""Small deterministic tools used by the inspection agent."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.db import SupabaseDB

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = PROJECT_ROOT / "demo_data" / "thresholds.json"
PROCEDURES_DIR = Path(__file__).resolve().parent / "procedures"


def _heatmap_hot_fraction(path: str | Path | None) -> float | None:
    """Estimate how much of a heatmap is strongly highlighted, when Pillow works."""
    try:
        from PIL import Image
    except ImportError:
        return None

    if not path:
        return None
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((256, 256))
            pixels = list(image.getdata())
    except (OSError, ValueError):
        return None

    if not pixels:
        return None
    # Red-heavy areas in the detector's JET heatmap are a rough visual signal.
    hot_pixels = sum(1 for red, green, blue in pixels if red > 120 and red > green * 1.25 and red > blue * 1.25)
    return hot_pixels / len(pixels)


def classify_defect(
    db: Any,
    score: float,
    heatmap_path: str | Path | None = None,
) -> dict[str, Any]:
    """Classify from score thresholds and heatmap content, never from path names."""
    del db  # The database is accepted consistently; classification itself is local.
    with THRESHOLDS_PATH.open(encoding="utf-8") as file:
        thresholds = json.load(file)
    high_threshold = float(thresholds["high_threshold"])
    if high_threshold <= 0:
        raise ValueError("The high threshold must be greater than zero.")

    severity_ratio = score / high_threshold
    if severity_ratio >= 1.5:
        severity = "high"
    elif severity_ratio >= 1.15:
        severity = "medium"
    else:
        severity = "low"

    hot_fraction = _heatmap_hot_fraction(heatmap_path)
    if hot_fraction is None:
        defect_type = "unspecified_anomaly"
        confidence = "medium"
    else:
        # Broad highlights suggest contamination; localized highlights suggest damage.
        defect_type = (
            "contamination"
            if hot_fraction >= 0.12
            else "broken_or_chipped_bottle"
        )
        # These are simple heatmap heuristics; connect a reviewed classifier here later.
        confidence = "medium"
        relative_gap = abs(score - high_threshold) / abs(high_threshold)
        if relative_gap <= 0.05:
            confidence = "low"

    # Future extension point: an approved local or hosted LLM can enrich this result.
    return {
        "defect_type": defect_type,
        "severity": severity,
        "confidence": confidence,
        "hot_fraction": hot_fraction,
    }


def fetch_procedure(db: Any, defect_type: str) -> str:
    """Find a short procedure by matching the fixed defect-type keywords."""
    del db
    keywords = {
        "broken": "broken_or_chipped_bottle.txt",
        "chipped": "broken_or_chipped_bottle.txt",
        "contamination": "contamination.txt",
        "unspecified": "unspecified_anomaly.txt",
        "general": "general_hold_and_inspect.txt",
    }
    normalized = defect_type.lower()
    filename = next(
        (name for word, name in keywords.items() if word in normalized),
        "general_hold_and_inspect.txt",
    )
    return (PROCEDURES_DIR / filename).read_text(encoding="utf-8")


def create_ticket(
    db: Any,
    inspection_id: Any,
    defect_type: str,
    severity: str,
    procedure_text: str,
) -> tuple[dict[str, Any], bool]:
    """Create one open ticket, unless this inspection already has one."""
    existing = db.list_tickets(inspection_id=inspection_id)
    if existing:
        return existing[0], False
    ticket = db.create_ticket(
        {
            "inspection_id": inspection_id,
            "defect_type": defect_type,
            "severity": severity,
            "procedure_text": procedure_text,
            "status": "open",
        }
    )
    return ticket, True


def request_review(
    db: Any,
    inspection_id: Any,
    reason: str,
    reviewer_label: str = "quality_control",
) -> tuple[dict[str, Any], bool]:
    """Create a review request only when this inspection has no review yet."""
    existing = db.list_reviews(inspection_id=inspection_id)
    if existing:
        return existing[0], False
    review = db.add_review(
        {
            "inspection_id": inspection_id,
            "reason": reason,
            "reviewer_label": reviewer_label,
            "notes": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reviewed_at": None,
        }
    )
    return review, True


def log_decision(
    db: Any,
    inspection_id: Any,
    step: str,
    tool_name: str,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """Record a short decision trace without storing secrets."""
    return db.log(
        {
            "inspection_id": inspection_id,
            "step": step,
            "tool_name": tool_name,
            "input_summary": input_summary[:500],
            "output_summary": output_summary[:500],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
