"""FastAPI endpoint tests using only the in-memory database."""

import pytest
from fastapi.testclient import TestClient

from agent.db import InMemoryDB
import backend.main as backend_main
from backend.main import create_app


@pytest.fixture
def database() -> InMemoryDB:
    db = InMemoryDB()
    broken = db.upsert_inspection(
        {
            "image_path": "bottle/test/good/000.png",
            "heatmap_path": "good/000.png",
            "score": 0.8,
            "decision": "defect",
        }
    )
    pending = db.upsert_inspection(
        {
            "image_path": "bottle/test/good/001.png",
            "heatmap_path": "good/001.png",
            "score": 0.6,
            "decision": "uncertain",
        }
    )
    unsafe = db.upsert_inspection(
        {
            "image_path": "../../README.md",
            "heatmap_path": "../thresholds.json",
            "score": 0.9,
            "decision": "defect",
        }
    )
    db.create_ticket(
        {
            "inspection_id": broken["id"],
            "defect_type": "broken_or_chipped_bottle",
            "severity": "high",
            "procedure_text": "Hold and inspect safely.",
            "status": "open",
        }
    )
    db.add_review(
        {
            "inspection_id": pending["id"],
            "reason": "Manual confirmation needed.",
            "reviewer_label": "",
            "notes": "",
            "reviewed_at": None,
        }
    )
    db.add_review(
        {
            "inspection_id": unsafe["id"],
            "reason": "Unsafe path fixture.",
            "reviewer_label": "",
            "notes": "",
            "reviewed_at": None,
        }
    )
    db.log(
        {
            "inspection_id": broken["id"],
            "step": "classify",
            "tool_name": "agent",
            "input_summary": "score",
            "output_summary": "defect",
            "created_at": "2026-01-01T00:00:00Z",
        }
    )
    db.log(
        {
            "inspection_id": broken["id"],
            "step": "ticket",
            "tool_name": "agent",
            "input_summary": "procedure",
            "output_summary": "created",
            "created_at": "2026-01-02T00:00:00Z",
        }
    )
    return db


@pytest.fixture
def client(database: InMemoryDB) -> TestClient:
    return TestClient(create_app(database))


def test_health_and_overview(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "offline": True}

    overview = client.get("/api/overview")
    assert overview.status_code == 200
    assert overview.json()["total_inspections"] == 3
    assert overview.json()["normal"] == 0
    assert overview.json()["uncertain"] == 1
    assert overview.json()["defect"] == 2
    assert overview.json()["open_tickets"] == 1
    assert overview.json()["pending_reviews"] == 2
    assert "low_threshold" in overview.json()["thresholds"]


def test_legacy_unreviewed_agent_placeholder_is_pending() -> None:
    db = InMemoryDB()
    inspection = db.upsert_inspection(
        {
            "image_path": "bottle/test/good/002.png",
            "heatmap_path": "good/002.png",
            "score": 0.6,
            "decision": "uncertain",
        }
    )
    db.add_review(
        {
            "inspection_id": inspection["id"],
            "reason": "Needs manual review.",
            "reviewer_label": "quality_control",
            "notes": "",
            "reviewed_at": None,
        }
    )
    client = TestClient(create_app(db))

    assert len(client.get("/api/reviews?pending=true").json()) == 1
    assert client.get("/api/overview").json()["pending_reviews"] == 1

    # The same legacy label is not pending after a reviewer timestamp exists.
    db.reviews[0]["reviewed_at"] = "2026-11-10T00:00:00Z"
    assert client.get("/api/reviews?pending=true").json() == []
    assert client.get("/api/overview").json()["pending_reviews"] == 0


def test_offline_mode_seeds_database_from_demo_csv(monkeypatch) -> None:
    monkeypatch.setenv("DASHBOARD_OFFLINE", "1")
    client = TestClient(create_app())
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["offline"] is True
    assert client.get("/api/overview").json()["total_inspections"] == 83


def test_inspections_list_filter_detail_and_pagination(client: TestClient) -> None:
    filtered = client.get("/api/inspections?decision=uncertain&limit=1&offset=0")
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert len(filtered.json()["items"]) == 1

    detail = client.get("/api/inspections/1")
    assert detail.status_code == 200
    assert detail.json()["ticket"]["id"] == detail.json()["tickets"][0]["id"]
    assert len(detail.json()["tickets"]) == 1
    assert [row["step"] for row in detail.json()["decision_log"]] == ["classify", "ticket"]
    assert client.get("/api/inspections/9999").status_code == 404


def test_ticket_filters_and_status_update(client: TestClient) -> None:
    tickets = client.get(
        "/api/tickets?status=open&severity=high&defect_type=broken_or_chipped_bottle"
    )
    assert tickets.status_code == 200
    assert len(tickets.json()) == 1

    updated = client.patch("/api/tickets/4", json={"status": "in_progress"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "in_progress"
    assert client.patch("/api/tickets/4", json={"status": "invalid"}).status_code == 422


def test_reviews_pending_and_resolve_defect_without_duplicate_ticket(
    client: TestClient,
    database: InMemoryDB,
) -> None:
    pending = client.get("/api/reviews?pending=true")
    assert pending.status_code == 200
    assert len(pending.json()) == 2

    first = client.post(
        "/api/reviews/5/resolve",
        json={"reviewer_label": "defect", "notes": "Confirmed on the line."},
    )
    assert first.status_code == 200
    assert first.json()["review"]["reviewer_label"] == "defect"
    assert first.json()["review"]["reviewed_at"]
    assert first.json()["ticket"] is not None
    ticket_count = len(database.list_tickets())

    second = client.post(
        "/api/reviews/5/resolve",
        json={"reviewer_label": "defect", "notes": "Confirmed again."},
    )
    assert second.status_code == 200
    assert len(database.list_tickets()) == ticket_count
    assert client.get("/api/reviews?pending=true").json().__len__() == 1

    normal = client.post(
        "/api/reviews/6/resolve",
        json={"reviewer_label": "normal", "notes": "Looks acceptable."},
    )
    assert normal.status_code == 200
    assert normal.json()["ticket"] is None
    assert client.post(
        "/api/reviews/999/resolve",
        json={"reviewer_label": "normal", "notes": ""},
    ).status_code == 404


def test_report_json_and_html_include_inspection_id(client: TestClient) -> None:
    report = client.get("/api/reports/4")
    assert report.status_code == 200
    inspection_id = report.json()["inspection_id"]

    html_report = client.get("/api/reports/4/html")
    assert html_report.status_code == 200
    assert html_report.headers["content-type"].startswith("text/html")
    assert f"Inspection ID:</strong> {inspection_id}" in html_report.text
    assert "data:image/png;base64," in html_report.text
    assert "Corrective procedure" in html_report.text
    assert "Decision log" in html_report.text


def test_image_streams_and_path_traversal_is_rejected(client: TestClient) -> None:
    original = client.get("/api/images/original/1")
    heatmap = client.get("/api/images/heatmap/1")
    assert original.status_code == 200
    assert original.headers["content-type"] == "image/png"
    assert heatmap.status_code == 200
    assert heatmap.headers["content-type"] == "image/png"

    unsafe_original = client.get("/api/images/original/3")
    unsafe_heatmap = client.get("/api/images/heatmap/3")
    assert unsafe_original.status_code == 404
    assert unsafe_heatmap.status_code == 404


def test_backend_serves_frontend_index_and_fallback(monkeypatch, tmp_path) -> None:
    (tmp_path / "index.html").write_text(
        "<!doctype html><title>InspectAI test frontend</title>",
        encoding="utf-8",
    )
    monkeypatch.setattr(backend_main, "FRONTEND_DIST", tmp_path)
    client = TestClient(create_app(InMemoryDB()))

    assert "InspectAI test frontend" in client.get("/").text
    assert "InspectAI test frontend" in client.get("/inspections").text
