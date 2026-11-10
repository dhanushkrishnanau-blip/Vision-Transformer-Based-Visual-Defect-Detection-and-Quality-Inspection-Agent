"""FastAPI endpoints for the React quality dashboard."""

import csv
import html
import json
import mimetypes
import os
from base64 import b64encode
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from agent.db import InMemoryDB, SupabaseDB, SupabaseError
from agent.graph import run_detection
from agent.run_agent import _clean_path
from agent.tools import classify_defect, create_ticket, fetch_procedure, log_decision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "demo_data"
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec"
HEATMAP_ROOT = DEMO_ROOT / "heatmaps"
THRESHOLDS_PATH = DEMO_ROOT / "thresholds.json"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


class TicketStatusUpdate(BaseModel):
    """A ticket status accepted by the update endpoint."""

    status: Literal["open", "in_progress", "closed"]


class ReviewResolution(BaseModel):
    """A human reviewer decision and optional notes."""

    reviewer_label: Literal["defect", "normal"]
    notes: str = Field(default="", max_length=4000)


def _seed_offline_database() -> InMemoryDB:
    """Replay the existing demo CSV through the agent into memory."""
    database = InMemoryDB()
    csv_path = DEMO_ROOT / "detections.csv"
    if not csv_path.is_file():
        raise FileNotFoundError("Offline detections file is missing.")
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "heatmap_path", "score", "decision"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Offline detections file is missing required columns.")
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


def _get_database(request: Request) -> Any:
    """Return the app's configured database, creating it once when needed."""
    database = request.app.state.database
    if database is None:
        if os.getenv("DASHBOARD_OFFLINE", "").strip() == "1":
            database = _seed_offline_database()
        else:
            database = SupabaseDB()
        request.app.state.database = database
    return database


def _same_id(left: Any, right: Any) -> bool:
    return str(left) == str(right)


def _find_inspection(database: Any, inspection_id: Any) -> dict[str, Any]:
    for row in database.list_inspections():
        if _same_id(row.get("id"), inspection_id):
            return row
    raise HTTPException(status_code=404, detail="Inspection was not found.")


def _find_review(database: Any, review_id: Any) -> dict[str, Any]:
    for row in database.list_reviews():
        if _same_id(row.get("id"), review_id):
            return row
    raise HTTPException(status_code=404, detail="Review was not found.")


def _find_ticket(database: Any, ticket_id: Any) -> dict[str, Any]:
    for row in database.list_tickets():
        if _same_id(row.get("id"), ticket_id):
            return row
    raise HTTPException(status_code=404, detail="Ticket was not found.")


def _rows_for_inspection(rows: list[dict[str, Any]], inspection_id: Any) -> list[dict[str, Any]]:
    return [row for row in rows if _same_id(row.get("inspection_id"), inspection_id)]


def _relative_asset_path(stored_path: str, marker: tuple[str, ...]) -> Path:
    """Extract a relative asset path and reject traversal or outside absolute paths."""
    normalized = stored_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    lowered = [part.lower() for part in parts]
    marker_lower = [part.lower() for part in marker]

    marker_index = next(
        (
            index
            for index in range(len(parts) - len(marker) + 1)
            if lowered[index : index + len(marker)] == marker_lower
        ),
        None,
    )
    if marker_index is not None:
        parts = parts[marker_index + len(marker) :]
    elif normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        raise HTTPException(status_code=404, detail="Image path is outside the allowed folder.")

    if not parts or any(part == ".." for part in parts):
        raise HTTPException(status_code=404, detail="Image path is outside the allowed folder.")

    base = DATA_ROOT if marker == ("data", "mvtec") else HEATMAP_ROOT
    candidate = (base / Path(*parts)).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Image path is outside the allowed folder.") from None
    return candidate


def _original_path(inspection: dict[str, Any]) -> Path:
    return _relative_asset_path(str(inspection.get("image_path", "")), ("data", "mvtec"))


def _heatmap_path(inspection: dict[str, Any]) -> Path:
    return _relative_asset_path(
        str(inspection.get("heatmap_path", "")),
        ("demo_data", "heatmaps"),
    ) if "demo_data/heatmaps" in str(inspection.get("heatmap_path", "")).replace("\\", "/") else _relative_asset_path(
        str(inspection.get("heatmap_path", "")),
        ("heatmaps",),
    )


def _read_thresholds() -> dict[str, float]:
    try:
        with THRESHOLDS_PATH.open(encoding="utf-8") as file:
            values = json.load(file)
        return {
            "low_threshold": float(values["low_threshold"]),
            "high_threshold": float(values["high_threshold"]),
        }
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(status_code=503, detail="Detector thresholds are unavailable.") from None


def _report_data(database: Any, ticket_id: Any) -> dict[str, Any]:
    ticket = _find_ticket(database, ticket_id)
    inspection = _find_inspection(database, ticket.get("inspection_id"))
    logs = _rows_for_inspection(
        database.list_log(), inspection.get("id")
    )
    logs.sort(key=lambda row: (str(row.get("created_at", "")), str(row.get("id", ""))))
    return {
        "inspection_id": inspection.get("id"),
        "inspection": inspection,
        "ticket": ticket,
        "decision_log": logs,
    }


def _embedded_image(path: Path, label: str) -> str:
    """Embed a safe image as a data URL for downloadable standalone reports."""
    if not path.is_file():
        return f'<div class="missing">{html.escape(label)} is unavailable.</div>'
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<img alt="{html.escape(label, quote=True)}" '
        f'src="data:{media_type};base64,{encoded}" '
        'style="max-width:48%;max-height:520px;object-fit:contain">'
    )


def _report_html(report: dict[str, Any]) -> str:
    inspection = report["inspection"]
    ticket = report["ticket"]
    images = (
        _embedded_image(_original_path(inspection), "Original inspection image")
        + _embedded_image(_heatmap_path(inspection), "Anomaly heatmap")
    )
    logs = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('created_at', '')))}</td>"
        f"<td>{html.escape(str(row.get('step', '')))}</td>"
        f"<td>{html.escape(str(row.get('tool_name', '')))}</td>"
        f"<td>{html.escape(str(row.get('input_summary', '')))}</td>"
        f"<td>{html.escape(str(row.get('output_summary', '')))}</td></tr>"
        for row in report["decision_log"]
    ) or '<tr><td colspan="5">No decision log entries.</td></tr>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Inspection report {html.escape(str(report['inspection_id']))}</title>
<style>body{{font:16px Arial;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;width:100%;margin-top:1rem}}th,td{{border:1px solid #bbb;padding:.5rem;text-align:left;overflow-wrap:anywhere}}
.images{{display:flex;gap:1rem;align-items:flex-start}}.missing{{border:1px dashed #888;padding:2rem}}</style></head>
<body><h1>Quality inspection report</h1>
<p><strong>Inspection ID:</strong> {html.escape(str(report['inspection_id']))}</p>
<p><strong>Image path:</strong> {html.escape(str(inspection.get('image_path', '')))}</p>
<p><strong>Score:</strong> {html.escape(str(inspection.get('score', '')))}</p>
<p><strong>Decision:</strong> {html.escape(str(inspection.get('decision', '')))}</p>
<p><strong>Defect type:</strong> {html.escape(str(ticket.get('defect_type', '')))}</p>
<p><strong>Severity:</strong> {html.escape(str(ticket.get('severity', '')))}</p>
<h2>Images</h2><div class="images">{images}</div>
<h2>Corrective procedure</h2><pre>{html.escape(str(ticket.get('procedure_text', '')))}</pre>
<h2>Decision log</h2><p>All entries belong to inspection ID {html.escape(str(report['inspection_id']))}.</p>
<table><thead><tr><th>Time</th><th>Step</th><th>Tool</th><th>Input</th><th>Output</th></tr></thead>
<tbody>{logs}</tbody></table><button onclick="window.print()">Print</button></body></html>"""


def create_app(database: Any = None) -> FastAPI:
    """Create the API and optionally inject an in-memory database for tests."""
    application = FastAPI(title="InspectAI Dashboard API", version="1.0")
    application.state.database = database
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(SupabaseError)
    async def supabase_error_handler(_request: Request, _error: SupabaseError):
        return JSONResponse(
            status_code=503,
            content={"detail": "Database request failed. Check Supabase configuration and access."},
        )

    @application.exception_handler(Exception)
    async def safe_error_handler(_request: Request, _error: Exception):
        # Do not expose internal exceptions, request headers, or secrets to clients.
        return JSONResponse(
            status_code=500,
            content={"detail": "The server could not complete this request."},
        )

    @application.get("/api/health")
    def health(db: Any = Depends(_get_database)):
        return {"status": "ok", "offline": isinstance(db, InMemoryDB)}

    @application.get("/api/overview")
    def overview(db: Any = Depends(_get_database)):
        inspections = db.list_inspections()
        tickets = db.list_tickets()
        reviews = db.list_reviews()
        pending = [
            row for row in reviews
            if not str(row.get("reviewer_label") or "").strip()
        ]
        decisions = {
            name: sum(str(row.get("decision", "")).lower() == name for row in inspections)
            for name in ("normal", "uncertain", "defect")
        }
        reviewed_inspection_ids = {
            str(row.get("inspection_id")) for row in reviews
        }
        return {
            "total_inspections": len(inspections),
            **decisions,
            "open_tickets": sum(row.get("status") == "open" for row in tickets),
            "pending_reviews": len(pending),
            "cases_sent_to_review": len(reviewed_inspection_ids),
            "share_sent_to_review": (
                len(reviewed_inspection_ids) / len(inspections) if inspections else 0
            ),
            "thresholds": _read_thresholds(),
        }

    @application.get("/api/inspections")
    def inspections(
        decision: Literal["normal", "uncertain", "defect"] | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        db: Any = Depends(_get_database),
    ):
        rows = db.list_inspections(status=decision)
        rows.sort(key=lambda row: (str(row.get("created_at", "")), str(row.get("id", ""))))
        return {"items": rows[offset : offset + limit], "total": len(rows)}

    @application.get("/api/inspections/{inspection_id}")
    def inspection_detail(inspection_id: str, db: Any = Depends(_get_database)):
        row = _find_inspection(db, inspection_id)
        tickets = _rows_for_inspection(db.list_tickets(), row.get("id"))
        reviews = _rows_for_inspection(db.list_reviews(), row.get("id"))
        logs = _rows_for_inspection(db.list_log(), row.get("id"))
        logs.sort(key=lambda item: (str(item.get("created_at", "")), str(item.get("id", ""))))
        return {
            "inspection": row,
            "ticket": tickets[0] if tickets else None,
            "review": reviews[0] if reviews else None,
            "tickets": tickets,
            "reviews": reviews,
            "decision_log": logs,
        }

    @application.get("/api/tickets")
    def tickets(
        status: Literal["open", "in_progress", "closed"] | None = None,
        severity: str | None = None,
        defect_type: str | None = None,
        db: Any = Depends(_get_database),
    ):
        rows = db.list_tickets(status=status)
        return [
            row for row in rows
            if (severity is None or row.get("severity") == severity)
            and (defect_type is None or row.get("defect_type") == defect_type)
        ]

    @application.patch("/api/tickets/{ticket_id}")
    def update_ticket(
        ticket_id: str,
        update: TicketStatusUpdate,
        db: Any = Depends(_get_database),
    ):
        ticket = _find_ticket(db, ticket_id)
        return db.update_ticket_status(ticket["id"], update.status)

    @application.get("/api/reviews")
    def reviews(
        pending: bool = False,
        db: Any = Depends(_get_database),
    ):
        rows = db.list_reviews()
        if pending:
            rows = [
                row for row in rows
                if not str(row.get("reviewer_label") or "").strip()
            ]
        return rows

    @application.post("/api/reviews/{review_id}/resolve")
    def resolve_review(
        review_id: str,
        resolution: ReviewResolution,
        db: Any = Depends(_get_database),
    ):
        review = _find_review(db, review_id)
        inspection = _find_inspection(db, review.get("inspection_id"))

        ticket = None
        if resolution.reviewer_label == "defect":
            # Use the agent's classification and procedure tools, not folder labels.
            heatmap = _heatmap_path(inspection)
            classification = classify_defect(
                db,
                float(inspection.get("score", 0)),
                heatmap if heatmap.is_file() else None,
            )
            procedure = fetch_procedure(db, classification["defect_type"])
            ticket, _ = create_ticket(
                db,
                inspection["id"],
                classification["defect_type"],
                classification["severity"],
                procedure,
            )

        resolved = db.resolve_review(
            review["id"],
            resolution.reviewer_label,
            resolution.notes,
        )
        log_decision(
            db,
            inspection["id"],
            step="manual_review",
            tool_name="reviewer",
            input_summary=f"review_id={review['id']}",
            output_summary=f"{resolution.reviewer_label}: {resolution.notes}",
        )
        return {"review": resolved, "ticket": ticket}

    @application.get("/api/reports/{ticket_id}")
    def report(ticket_id: str, db: Any = Depends(_get_database)):
        return _report_data(db, ticket_id)

    @application.get("/api/reports/{ticket_id}/html", response_class=HTMLResponse)
    def report_html(ticket_id: str, db: Any = Depends(_get_database)):
        report_content = _report_html(_report_data(db, ticket_id))
        return HTMLResponse(
            report_content,
            headers={
                "Content-Disposition": 'attachment; filename="inspection-report.html"'
            },
        )

    def _image_response(inspection_id: str, heatmap: bool, db: Any):
        row = _find_inspection(db, inspection_id)
        path = _heatmap_path(row) if heatmap else _original_path(row)
        if not path.is_file():
            label = "Heatmap" if heatmap else "Original image"
            raise HTTPException(status_code=404, detail=f"{label} file was not found.")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @application.get("/api/images/original/{inspection_id}")
    def original_image(inspection_id: str, db: Any = Depends(_get_database)):
        return _image_response(inspection_id, False, db)

    @application.get("/api/images/heatmap/{inspection_id}")
    def heatmap_image(inspection_id: str, db: Any = Depends(_get_database)):
        return _image_response(inspection_id, True, db)

    if FRONTEND_DIST.is_dir():
        @application.get("/", include_in_schema=False)
        def frontend_index():
            index_file = FRONTEND_DIST / "index.html"
            if not index_file.is_file():
                raise HTTPException(status_code=404, detail="Frontend build is missing index.html.")
            return FileResponse(index_file)

        @application.get("/{frontend_path:path}", include_in_schema=False)
        def frontend_fallback(frontend_path: str):
            if frontend_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API endpoint was not found.")
            root = FRONTEND_DIST.resolve()
            target = (root / frontend_path).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                raise HTTPException(status_code=404, detail="Frontend file was not found.") from None
            if target.is_file():
                return FileResponse(target)
            index_file = root / "index.html"
            if index_file.is_file():
                return FileResponse(index_file)
            raise HTTPException(status_code=404, detail="Frontend build was not found.")
    else:
        @application.get("/", include_in_schema=False)
        def frontend_not_built():
            return JSONResponse(
                status_code=404,
                content={"detail": "React frontend build not found. Start Vite on port 5173."},
            )

    return application


app = create_app()
