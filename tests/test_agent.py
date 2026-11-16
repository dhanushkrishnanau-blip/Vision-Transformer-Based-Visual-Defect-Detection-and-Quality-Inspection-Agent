"""Offline tests for the inspection workflow; these never contact Supabase."""

import json
from pathlib import Path

import pytest

from agent import graph, llm
from agent.db import InMemoryDB
from agent.graph import run_detection
from agent.tools import classify_defect


@pytest.fixture(autouse=True)
def mock_ollama_for_existing_tests(monkeypatch):
    monkeypatch.setattr(
        graph,
        "classify_inspection",
        lambda metadata: {
            "defect_type": metadata["rule_defect_type"],
            "rationale": "Mocked subtype refinement.",
        },
    )


class FakeOllamaResponse:
    def __init__(self, content: str) -> None:
        self.ok = True
        self.status_code = 200
        self.content = content

    def json(self) -> dict[str, object]:
        return {"message": {"content": self.content}}


def use_mocked_ollama(monkeypatch, content: str | None = None, error=None):
    monkeypatch.setattr(graph, "classify_inspection", llm.classify_inspection)
    monkeypatch.setattr(llm, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3:1.7b")
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if error is not None:
            raise error
        return FakeOllamaResponse(content or json.dumps(
            {
                "defect_type": "contamination",
                "rationale": "The provided rule evidence supports contamination.",
            }
        ))

    monkeypatch.setattr(llm.requests, "post", fake_post)
    return calls


def set_rule_result(
    monkeypatch,
    *,
    defect_type: str = "unspecified_anomaly",
    severity: str = "high",
    confidence: str = "medium",
    hot_fraction: float | None = None,
) -> None:
    monkeypatch.setattr(
        graph,
        "classify_defect",
        lambda db, score, heatmap_path: {
            "defect_type": defect_type,
            "severity": severity,
            "confidence": confidence,
            "hot_fraction": hot_fraction,
            "low_threshold": 0.3,
            "high_threshold": 0.5,
        },
    )


def test_normal_inspection_skips_ollama(monkeypatch) -> None:
    def unexpected_call(metadata):
        raise AssertionError("Normal decisions must not call Ollama.")

    monkeypatch.setattr(graph, "classify_inspection", unexpected_call)

    result = run_detection(InMemoryDB(), detection("normal-no-llm", "normal", 0.1))

    assert result["decision"] == "normal"
    assert not result.get("ticket_created")


def test_uncertain_inspection_skips_ollama_and_requests_review(monkeypatch) -> None:
    def unexpected_call(metadata):
        raise AssertionError("Uncertain decisions must not call Ollama.")

    monkeypatch.setattr(graph, "classify_inspection", unexpected_call)
    db = InMemoryDB()

    result = run_detection(db, detection("uncertain-no-llm", "uncertain"))

    assert result["review_created"] is True
    assert db.reviews[0]["reason"] == "Detector decision: uncertain"


def test_low_confidence_defect_skips_ollama_and_requests_review(monkeypatch) -> None:
    set_rule_result(monkeypatch, confidence="low")

    def unexpected_call(metadata):
        raise AssertionError("Low-confidence classifications must not call Ollama.")

    monkeypatch.setattr(graph, "classify_inspection", unexpected_call)
    db = InMemoryDB()

    result = run_detection(db, detection("low-confidence-no-llm", "defect"))

    assert result["review_created"] is True
    assert db.reviews[0]["reason"] == "Classification confidence is low."
    assert db.tickets == []


def test_eligible_defect_calls_ollama_with_rule_evidence(monkeypatch) -> None:
    calls = use_mocked_ollama(monkeypatch)
    db = InMemoryDB()

    result = run_detection(db, detection("eligible-llm", "defect", 0.8))

    assert len(calls) == 1
    url, request = calls[0]
    sent = json.loads(request["json"]["messages"][1]["content"])
    assert url == llm.OLLAMA_API_URL
    assert request["json"]["model"] == "qwen3:1.7b"
    assert sent == {
        "decision": "defect",
        "score": 0.8,
        "rule_defect_type": "unspecified_anomaly",
        "rule_severity": "high",
        "rule_confidence": "medium",
        "hot_fraction": None,
        "low_threshold": 0.3,
        "high_threshold": 0.5,
    }
    assert result["defect_type"] == "contamination"
    assert db.tickets[0]["defect_type"] == "contamination"


def test_contradictory_subtype_routes_to_review(monkeypatch) -> None:
    set_rule_result(monkeypatch, defect_type="contamination")
    use_mocked_ollama(
        monkeypatch,
        json.dumps(
            {
                "defect_type": "broken_or_chipped_bottle",
                "rationale": "Contradicts the rule classification.",
            }
        ),
    )
    db = InMemoryDB()

    result = run_detection(db, detection("contradictory-llm", "defect"))

    assert result["review_created"] is True
    assert "unsupported by the rule evidence" in db.reviews[0]["reason"]
    assert db.tickets == []


@pytest.mark.parametrize("defect_type", ["scratched_bottle", "normal"])
def test_unsupported_subtype_routes_to_review(monkeypatch, defect_type) -> None:
    use_mocked_ollama(
        monkeypatch,
        json.dumps(
            {"defect_type": defect_type, "rationale": "Unsupported subtype."}
        ),
    )
    db = InMemoryDB()

    result = run_detection(db, detection("unsupported-llm", "defect"))

    assert result["review_created"] is True
    assert "unsupported defect type" in db.reviews[0]["reason"]
    assert db.tickets == []


def test_malformed_ollama_json_routes_to_review(monkeypatch) -> None:
    use_mocked_ollama(monkeypatch, "not json")
    db = InMemoryDB()

    result = run_detection(db, detection("malformed-llm", "defect"))

    assert result["review_created"] is True
    assert "not valid JSON" in db.reviews[0]["reason"]
    assert any(
        row["tool_name"] == "agent_error"
        and "not valid JSON" in row["output_summary"]
        for row in db.logs
    )
    error_input = next(
        row["input_summary"] for row in db.logs if row["tool_name"] == "agent_error"
    )
    assert "decision=defect; score=0.8" in error_input
    assert "rule_defect_type=unspecified_anomaly" in error_input
    assert "severity=high" in error_input
    assert "high_threshold=0.5" in error_input
    assert db.tickets == []


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (llm.requests.ConnectionError("offline"), "request failed"),
        (llm.requests.Timeout("slow"), "timed out"),
    ],
)
def test_ollama_request_failures_route_to_review(
    monkeypatch, failure, message
) -> None:
    use_mocked_ollama(monkeypatch, error=failure)
    db = InMemoryDB()

    result = run_detection(db, detection("ollama-failure", "defect"))

    assert result["review_created"] is True
    assert message in db.reviews[0]["reason"]
    assert db.tickets == []


def test_invalid_model_confidence_field_routes_to_review(monkeypatch) -> None:
    use_mocked_ollama(
        monkeypatch,
        json.dumps(
            {
                "defect_type": "contamination",
                "rationale": "A subtype.",
                "confidence": "certain",
            }
        ),
    )
    db = InMemoryDB()

    result = run_detection(db, detection("invalid-confidence-llm", "defect"))

    assert result["review_created"] is True
    assert "only defect_type and rationale" in db.reviews[0]["reason"]
    assert db.tickets == []


def test_invalid_rule_confidence_routes_to_review_without_request(monkeypatch) -> None:
    set_rule_result(monkeypatch, confidence="certain")
    calls = []

    def unexpected_post(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Invalid rule confidence must be rejected before request.")

    monkeypatch.setattr(llm, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm.requests, "post", unexpected_post)
    monkeypatch.setattr(graph, "classify_inspection", llm.classify_inspection)
    db = InMemoryDB()

    result = run_detection(db, detection("invalid-rule-confidence", "defect"))

    assert result["review_created"] is True
    assert "invalid confidence" in db.reviews[0]["reason"]
    assert calls == []
    assert db.tickets == []


def test_detector_score_and_rule_severity_are_preserved(monkeypatch) -> None:
    set_rule_result(monkeypatch, severity="medium")
    use_mocked_ollama(monkeypatch)
    db = InMemoryDB()
    row = detection("evidence-preserved", "defect", 0.8)

    result = run_detection(db, row)

    assert result["decision"] == "defect"
    assert result["score"] == 0.8
    assert result["severity"] == "medium"
    assert db.inspections[0]["decision"] == "defect"
    assert db.inspections[0]["score"] == 0.8
    assert db.tickets[0]["severity"] == "medium"


def test_procedure_lookup_receives_validated_subtype(monkeypatch) -> None:
    use_mocked_ollama(monkeypatch)
    procedure_types = []

    def fake_procedure(db, defect_type):
        procedure_types.append(defect_type)
        return "Validated procedure."

    monkeypatch.setattr(graph, "fetch_procedure", fake_procedure)
    db = InMemoryDB()

    result = run_detection(db, detection("validated-procedure", "defect"))

    assert procedure_types == ["contamination"]
    assert result["defect_type"] == "contamination"
    assert db.tickets[0]["procedure_text"] == "Validated procedure."


def test_repeated_ollama_failure_does_not_duplicate_reviews(monkeypatch) -> None:
    use_mocked_ollama(monkeypatch, "not json")
    db = InMemoryDB()
    row = detection("duplicate-review-llm", "defect")

    first = run_detection(db, row)
    second = run_detection(db, row)

    assert first["review_created"] is True
    assert second["review_created"] is False
    assert len(db.reviews) == 1
    assert db.tickets == []


def test_repeated_valid_refinement_does_not_duplicate_tickets(monkeypatch) -> None:
    use_mocked_ollama(monkeypatch)
    db = InMemoryDB()
    row = detection("duplicate-ticket-llm", "defect")

    first = run_detection(db, row)
    second = run_detection(db, row)

    assert first["ticket_created"] is True
    assert second["ticket_created"] is False
    assert len(db.tickets) == 1
    assert db.reviews == []


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
