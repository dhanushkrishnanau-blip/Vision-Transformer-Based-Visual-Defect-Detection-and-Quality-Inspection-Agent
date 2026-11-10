"""Offline tests for the inspection workflow; these never contact Supabase."""

from pathlib import Path

from agent.db import InMemoryDB
from agent.graph import run_detection
from agent.tools import classify_defect


def detection(name: str, decision: str, score: float = 0.8) -> dict[str, str]:
    return {
        "image_path": f"bottle/test/{name}.png",
        "heatmap_path": f"{name}.png",
        "score": str(score),
        "decision": decision,
    }


def test_defect_creates_ticket() -> None:
    db = InMemoryDB()
    result = run_detection(db, detection("defect", "defect"))
    assert result["ticket_created"] is True
    assert len(db.tickets) == 1
    assert db.tickets[0]["status"] == "open"


def test_normal_creates_no_ticket_or_review() -> None:
    db = InMemoryDB()
    run_detection(db, detection("normal", "normal", 0.1))
    assert db.tickets == []
    assert db.reviews == []
    assert len(db.logs) == 1


def test_uncertain_requests_review() -> None:
    db = InMemoryDB()
    result = run_detection(db, detection("uncertain", "uncertain"))
    assert result["review_created"] is True
    assert len(db.reviews) == 1
    assert db.tickets == []


def test_tool_error_requests_review(monkeypatch) -> None:
    db = InMemoryDB()

    def fail_procedure(*args, **kwargs):
        raise RuntimeError("procedure lookup failed")

    monkeypatch.setattr("agent.graph.fetch_procedure", fail_procedure)
    result = run_detection(db, detection("error", "defect"))
    assert result["review_created"] is True
    assert "procedure lookup failed" in result["error"]
    assert db.tickets == []
    assert len(db.reviews) == 1
    assert any(row["tool_name"] == "agent_error" for row in db.logs)


def test_running_twice_does_not_duplicate_ticket_or_review() -> None:
    db = InMemoryDB()
    row = detection("same-defect", "defect")
    run_detection(db, row)
    second = run_detection(db, row)
    assert second["ticket_created"] is False
    assert len(db.inspections) == 1
    assert len(db.tickets) == 1

    review_row = detection("same-review", "uncertain")
    run_detection(db, review_row)
    review_second = run_detection(db, review_row)
    assert review_second["review_created"] is False
    assert len(db.reviews) == 1


def test_classification_does_not_use_path_name() -> None:
    db = InMemoryDB()
    first = classify_defect(db, 0.8, Path("broken_large") / "image.png")
    second = classify_defect(db, 0.8, Path("contamination") / "image.png")
    assert first == second
