"""Data helpers shared by the dashboard and its offline tests."""

import base64
import csv
import html
import json
import mimetypes
from pathlib import Path
from typing import Any

from agent.db import InMemoryDB
from agent.graph import run_detection
from agent.run_agent import _clean_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_PATH = PROJECT_ROOT / "demo_data" / "detections.csv"
THRESHOLDS_PATH = PROJECT_ROOT / "demo_data" / "thresholds.json"


def read_thresholds() -> dict[str, float]:
    """Read the detector's saved thresholds, with a clear missing-file error."""
    with THRESHOLDS_PATH.open(encoding="utf-8") as file:
        thresholds = json.load(file)
    return {
        "low_threshold": float(thresholds["low_threshold"]),
        "high_threshold": float(thresholds["high_threshold"]),
    }


def seed_offline_db() -> InMemoryDB:
    """Create demo rows in memory by replaying the agent's detection CSV."""
    database = InMemoryDB()
    with DETECTIONS_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "heatmap_path", "score", "decision"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("detections.csv is missing required columns.")
        for row in reader:
            detection = {
                "image_path": _clean_path(row["image_path"], "data/mvtec"),
                "heatmap_path": _clean_path(row["heatmap_path"], "heatmaps"),
                "score": row["score"],
                "decision": row["decision"].strip().lower(),
                "timestamp": row.get("timestamp", ""),
            }
            run_detection(database, detection)
    return database


def resolve_image_path(image_path: str) -> Path:
    """Resolve the normalized inspection path to its local original image."""
    safe_parts = [part for part in Path(image_path.replace("\\", "/")).parts if part not in {"..", "/"}]
    return PROJECT_ROOT / "data" / "mvtec" / Path(*safe_parts)


def resolve_heatmap_path(heatmap_path: str) -> Path:
    """Resolve the normalized heatmap path under demo_data/heatmaps."""
    safe_parts = [part for part in Path(heatmap_path.replace("\\", "/")).parts if part not in {"..", "/"}]
    return PROJECT_ROOT / "demo_data" / "heatmaps" / Path(*safe_parts)


def image_markup(path: Path, label: str, *, max_width: str = "100%") -> str:
    """Return a base64 image tag, or a readable placeholder when unavailable."""
    if not path.is_file():
        return (
            '<div class="image-placeholder">'
            f"{html.escape(label)} is unavailable<br>"
            f"<small>{html.escape(str(path))}</small></div>"
        )
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<img alt="{html.escape(label, quote=True)}" '
        f'src="data:{content_type};base64,{encoded}" '
        f'style="max-width:{max_width};max-height:420px;object-fit:contain">'
    )


def build_report_html(
    inspection: dict[str, Any],
    ticket: dict[str, Any],
    logs: list[dict[str, Any]],
) -> str:
    """Build a standalone printable report with its images embedded."""
    original = image_markup(resolve_image_path(str(inspection.get("image_path", ""))), "Original image")
    heatmap = image_markup(resolve_heatmap_path(str(inspection.get("heatmap_path", ""))), "Heatmap")
    ordered_logs = sorted(
        logs,
        key=lambda row: (str(row.get("created_at", "")), str(row.get("id", ""))),
    )
    log_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('created_at', '')))}</td>"
        f"<td>{html.escape(str(row.get('step', '')))}</td>"
        f"<td>{html.escape(str(row.get('tool_name', '')))}</td>"
        f"<td>{html.escape(str(row.get('input_summary', '')))}</td>"
        f"<td>{html.escape(str(row.get('output_summary', '')))}</td>"
        "</tr>"
        for row in ordered_logs
    ) or '<tr><td colspan="5">No decision log entries are available.</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inspection report - {html.escape(str(inspection.get('id', '')))}</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
h1,h2{{color:#17324d}}.images{{display:flex;gap:1rem;align-items:flex-start}}
.images figure{{flex:1;margin:0}}img{{width:100%;height:auto}}.image-placeholder{{border:1px dashed #888;padding:2rem}}
table{{border-collapse:collapse;width:100%;margin-top:1rem}}th,td{{border:1px solid #bbb;padding:.5rem;text-align:left;overflow-wrap:anywhere}}
@media print{{button{{display:none}}body{{margin:0}}}}
</style>
</head>
<body>
<h1>Quality inspection report</h1>
<p><strong>Inspection ID:</strong> {html.escape(str(inspection.get('id', '')))}</p>
<p><strong>Ticket ID:</strong> {html.escape(str(ticket.get('id', '')))}</p>
<p><strong>Image path:</strong> {html.escape(str(inspection.get('image_path', '')))}</p>
<p><strong>Detector decision:</strong> {html.escape(str(inspection.get('decision', '')))}</p>
<p><strong>Score:</strong> {html.escape(str(inspection.get('score', '')))}</p>
<p><strong>Defect type:</strong> {html.escape(str(ticket.get('defect_type', '')))}</p>
<p><strong>Severity:</strong> {html.escape(str(ticket.get('severity', '')))}</p>
<p><strong>Ticket status:</strong> {html.escape(str(ticket.get('status', '')))}</p>
<h2>Images</h2>
<div class="images"><figure><figcaption>Original</figcaption>{original}</figure>
<figure><figcaption>Heatmap</figcaption>{heatmap}</figure></div>
<h2>Corrective procedure</h2>
<pre>{html.escape(str(ticket.get('procedure_text', '')))}</pre>
<h2>Decision log</h2>
<p>Log entries are linked to inspection ID {html.escape(str(inspection.get('id', '')))}.</p>
<table><thead><tr><th>Time</th><th>Step</th><th>Tool</th><th>Input</th><th>Output</th></tr></thead>
<tbody>{log_rows}</tbody></table>
<button onclick="window.print()">Print report</button>
</body>
</html>"""


def ordered_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort a decision trail oldest first."""
    return sorted(
        logs,
        key=lambda row: (str(row.get("created_at", "")), str(row.get("id", ""))),
    )
