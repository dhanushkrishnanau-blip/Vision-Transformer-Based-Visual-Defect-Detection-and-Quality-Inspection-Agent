"""Offline tests for dashboard data preparation and report output."""

from pathlib import Path

from agent.db import InMemoryDB
from app.dashboard_data import build_report_html, seed_offline_db
from streamlit.testing.v1 import AppTest


def test_offline_seed_loads_all_demo_detections() -> None:
    db = seed_offline_db()
    assert len(db.list_inspections()) == 83
    assert len(db.list_tickets()) == 61
    assert len(db.list_reviews()) == 1


def test_report_contains_inspection_ticket_and_ordered_log_trail() -> None:
    db = InMemoryDB()
    inspection = db.upsert_inspection(
        {
            "image_path": "__missing_dashboard_test__/original.png",
            "heatmap_path": "__missing_dashboard_test__/heatmap.png",
            "score": 0.75,
            "decision": "defect",
        }
    )
    ticket = db.create_ticket(
        {
            "inspection_id": inspection["id"],
            "defect_type": "unspecified_anomaly",
            "severity": "medium",
            "procedure_text": "Hold and inspect.",
            "status": "open",
        }
    )
    later = {
        "id": 2,
        "inspection_id": inspection["id"],
        "created_at": "2026-01-02T00:00:00Z",
        "step": "ticket",
        "tool_name": "agent",
        "input_summary": "second",
        "output_summary": "ticket created",
    }
    earlier = {
        "id": 1,
        "inspection_id": inspection["id"],
        "created_at": "2026-01-01T00:00:00Z",
        "step": "classify",
        "tool_name": "agent",
        "input_summary": "first",
        "output_summary": "classified",
    }

    report = build_report_html(inspection, ticket, [later, earlier])

    assert "<!doctype html>" in report
    assert f"Inspection ID:</strong> {inspection['id']}" in report
    assert f"Ticket ID:</strong> {ticket['id']}" in report
    assert "Hold and inspect." in report
    assert report.index("2026-01-01") < report.index("2026-01-02")
    assert "Original image is unavailable" in report
    assert "Heatmap is unavailable" in report


def test_streamlit_pages_render_without_exceptions(monkeypatch) -> None:
    monkeypatch.setenv("DASHBOARD_OFFLINE", "1")
    dashboard = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py"),
        default_timeout=90,
    ).run()
    assert not dashboard.exception

    for page in ("Overview", "Inspections", "Tickets", "Review queue", "Reports"):
        dashboard.sidebar.radio[0].set_value(page).run()
        assert not dashboard.exception, f"{page} page raised an exception"
