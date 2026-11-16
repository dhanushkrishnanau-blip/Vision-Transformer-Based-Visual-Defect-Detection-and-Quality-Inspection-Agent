"""Mocked tests for the Ollama client; these never contact an API."""

import json

import pytest
import requests

from agent import llm


@pytest.fixture(autouse=True)
def ollama_environment(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", _no_dotenv)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3:1.7b")
    monkeypatch.delenv("OLLAMA_API_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)


def _no_dotenv(*args, **kwargs) -> None:
    _ = args, kwargs


def inspection_metadata() -> dict[str, object]:
    return {
        "decision": "defect",
        "score": 0.8,
        "rule_defect_type": "unspecified_anomaly",
        "rule_severity": "medium",
        "rule_confidence": "medium",
        "hot_fraction": 0.08,
        "image_path": "must-not-be-sent.png",
        "inspection_id": "must-not-be-sent",
    }


class FakeResponse:
    def __init__(self, body, *, status_code=200):
        self.body = body
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def mocked_post(response: FakeResponse):
    def post(*args, **kwargs):
        _ = args, kwargs
        return response

    return post


def ollama_response(content: str) -> dict[str, object]:
    return {"message": {"content": content}}


def result_json(
    defect_type: str = "unspecified_anomaly",
    rationale: str = "The available metadata does not support a more specific subtype.",
) -> str:
    return json.dumps({"defect_type": defect_type, "rationale": rationale})


def test_valid_response_sends_ollama_payload_and_allowlisted_metadata(
    monkeypatch,
) -> None:
    call = {}
    rationale = "The available metadata does not support a more specific subtype."

    def fake_post(url, **kwargs):
        call.update(url=url, **kwargs)
        return FakeResponse(
            ollama_response(result_json("unspecified_anomaly", rationale))
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)

    result = llm.classify_inspection(inspection_metadata())

    assert result == {
        "defect_type": "unspecified_anomaly",
        "rationale": rationale,
    }
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["timeout"] == llm.REQUEST_TIMEOUT_SECONDS
    assert 0 < call["timeout"] < float("inf")
    assert call["headers"] == {"Content-Type": "application/json"}
    assert "Authorization" not in call["headers"]
    payload = call["json"]
    assert payload["model"] == "qwen3:1.7b"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["temperature"] == 0
    user_content = payload["messages"][1]["content"]
    assert "must-not-be-sent.png" not in user_content
    assert "must-not-be-sent" not in user_content
    assert set(json.loads(user_content)) == {
        "decision",
        "score",
        "rule_defect_type",
        "rule_severity",
        "rule_confidence",
        "hot_fraction",
    }
    assert "anomaly score alone" in payload["messages"][0]["content"]
    assert "unspecified_anomaly" in payload["messages"][0]["content"]


def test_default_model_is_qwen3_1_7b(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODEL")
    assert llm._client_settings() == ("qwen3:1.7b", llm.OLLAMA_API_URL)


@pytest.mark.parametrize(
    ("rule_defect_type", "response_defect_type"),
    [
        ("contamination", "contamination"),
        ("contamination", "unspecified_anomaly"),
    ],
)
def test_accepts_rule_supported_type_or_safe_fallback(
    monkeypatch, rule_defect_type, response_defect_type
) -> None:
    metadata = inspection_metadata()
    metadata["rule_defect_type"] = rule_defect_type
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(
            FakeResponse(
                ollama_response(result_json(response_defect_type))
            )
        ),
    )

    result = llm.classify_inspection(metadata)

    assert result["defect_type"] == response_defect_type


@pytest.mark.parametrize("defect_type", ["contamination", "unspecified_anomaly"])
def test_unspecified_rule_accepts_supported_subtype_or_fallback(
    monkeypatch, defect_type
) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(FakeResponse(ollama_response(result_json(defect_type)))),
    )

    result = llm.classify_inspection(inspection_metadata())

    assert result["defect_type"] == defect_type


def test_sends_only_validated_threshold_evidence(monkeypatch) -> None:
    call = {}

    def fake_post(url, **kwargs):
        call.update(url=url, **kwargs)
        return FakeResponse(ollama_response(result_json()))

    monkeypatch.setattr(llm.requests, "post", fake_post)
    metadata = inspection_metadata()
    metadata.update(low_threshold=0.3, high_threshold=0.5, inspection_id="private")

    llm.classify_inspection(metadata)

    user_metadata = json.loads(call["json"]["messages"][1]["content"])
    assert user_metadata["low_threshold"] == 0.3
    assert user_metadata["high_threshold"] == 0.5
    assert "inspection_id" not in user_metadata


@pytest.mark.parametrize(
    "api_url",
    [
        "http://ollama.example.com:11434/api/chat",
        "http://192.0.2.1:11434/api/chat",
    ],
)
def test_rejects_remote_ollama_endpoints_before_request(monkeypatch, api_url) -> None:
    monkeypatch.setenv("OLLAMA_API_URL", api_url)
    request_called = False

    def fake_post(*args, **kwargs):
        nonlocal request_called
        _ = args, kwargs
        request_called = True
        raise AssertionError("Remote Ollama endpoint must not receive a request.")

    monkeypatch.setattr(llm.requests, "post", fake_post)

    with pytest.raises(llm.LLMClientError, match="localhost address"):
        llm.classify_inspection(inspection_metadata())
    assert request_called is False


def test_accepts_loopback_ip_endpoint(monkeypatch) -> None:
    api_url = "http://127.0.0.1:11434/api/chat"
    monkeypatch.setenv("OLLAMA_API_URL", api_url)
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(FakeResponse(ollama_response(result_json()))),
    )

    assert llm._client_settings() == ("qwen3:1.7b", api_url)


def test_no_api_key_is_required(monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(FakeResponse(ollama_response(result_json()))),
    )

    assert llm.classify_inspection(inspection_metadata())["defect_type"] == (
        "unspecified_anomaly"
    )


def test_rejects_malformed_completion_json(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(FakeResponse(ollama_response("not json"))),
    )

    with pytest.raises(llm.LLMClientError, match="not valid JSON"):
        llm.classify_inspection(inspection_metadata())


def test_rejects_invalid_http_response_json(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(FakeResponse(ValueError("invalid"))),
    )

    with pytest.raises(llm.LLMClientError, match="invalid JSON response"):
        llm.classify_inspection(inspection_metadata())


def test_rejects_missing_response_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(
            FakeResponse(
                ollama_response(json.dumps({"defect_type": "contamination"}))
            )
        ),
    )

    with pytest.raises(llm.LLMClientError, match="only defect_type and rationale"):
        llm.classify_inspection(inspection_metadata())


def test_rejects_incorrect_response_field_types(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(
            FakeResponse(
                ollama_response(
                    json.dumps(
                        {
                            "defect_type": "unspecified_anomaly",
                            "rationale": ["not", "text"],
                        }
                    )
                )
            )
        ),
    )

    with pytest.raises(llm.LLMClientError, match="rationale"):
        llm.classify_inspection(inspection_metadata())


def test_rejects_unsupported_defect_type(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(
            FakeResponse(
                ollama_response(
                    result_json("scratched_bottle", "Unsupported subtype.")
                )
            )
        ),
    )

    with pytest.raises(llm.LLMClientError, match="unsupported defect type"):
        llm.classify_inspection(inspection_metadata())


def test_rejects_specific_type_not_supported_by_rule_evidence(monkeypatch) -> None:
    metadata = inspection_metadata()
    metadata["rule_defect_type"] = "contamination"
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(
            FakeResponse(
                ollama_response(
                    result_json("broken_or_chipped_bottle", "Unsupported by rule.")
                )
            )
        ),
    )

    with pytest.raises(llm.LLMClientError, match="unsupported by the rule evidence"):
        llm.classify_inspection(metadata)


def test_rejects_non_ollama_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    with pytest.raises(llm.LLMClientError, match="must be set to 'ollama'"):
        llm.classify_inspection(inspection_metadata())


def test_timeout_is_reported_without_request_details(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        _ = args, kwargs
        raise requests.Timeout("private detail")

    monkeypatch.setattr(llm.requests, "post", raise_timeout)

    with pytest.raises(llm.LLMClientError, match="timed out") as error:
        llm.classify_inspection(inspection_metadata())
    assert "private detail" not in str(error.value)


def test_connection_failure_is_reported_without_request_details(monkeypatch) -> None:
    def raise_connection_error(*args, **kwargs):
        _ = args, kwargs
        raise requests.ConnectionError("private detail")

    monkeypatch.setattr(llm.requests, "post", raise_connection_error)

    with pytest.raises(llm.LLMClientError, match="request failed") as error:
        llm.classify_inspection(inspection_metadata())
    assert "private detail" not in str(error.value)


def test_http_error_is_reported_without_response_body(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        mocked_post(FakeResponse({"error": "private detail"}, status_code=503)),
    )

    with pytest.raises(llm.LLMClientError, match="HTTP 503") as error:
        llm.classify_inspection(inspection_metadata())
    assert "private detail" not in str(error.value)
