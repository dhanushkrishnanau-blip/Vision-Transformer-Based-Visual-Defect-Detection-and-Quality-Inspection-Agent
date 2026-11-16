"""Small Ollama client for structured, metadata-only defect classification."""

import json
import ipaddress
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:1.7b"
REQUEST_TIMEOUT_SECONDS = 20
SUPPORTED_DEFECT_TYPES = frozenset(
    {
        "broken_or_chipped_bottle",
        "contamination",
        "general_hold_and_inspect",
        "unspecified_anomaly",
    }
)
_SEVERITIES = frozenset({"low", "medium", "high"})
_RESPONSE_FIELDS = frozenset({"defect_type", "rationale"})


class LLMClientError(RuntimeError):
    """Raised when LLM configuration, requests, or responses are invalid."""


def _client_settings() -> tuple[str, str]:
    """Load the project environment and return the configured model and URL."""
    load_dotenv(PROJECT_ROOT / ".env")
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()
    api_url = os.getenv("OLLAMA_API_URL", OLLAMA_API_URL).strip()

    if provider != "ollama":
        raise LLMClientError("LLM_PROVIDER must be set to 'ollama'.")
    if not model:
        raise LLMClientError("LLM_MODEL is missing.")
    if not api_url:
        raise LLMClientError("OLLAMA_API_URL is missing.")
    try:
        parsed_url = urlsplit(api_url)
        hostname = parsed_url.hostname
    except ValueError:
        raise LLMClientError("OLLAMA_API_URL must use a localhost address.") from None
    try:
        is_loopback_ip = (
            hostname is not None and ipaddress.ip_address(hostname).is_loopback
        )
    except ValueError:
        is_loopback_ip = False
    if (
        parsed_url.scheme != "http"
        or (hostname != "localhost" and not is_loopback_ip)
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise LLMClientError("OLLAMA_API_URL must use a localhost address.")
    return model, api_url


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMClientError(f"Inspection metadata field '{field}' must be numeric.")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise LLMClientError(
            f"Inspection metadata field '{field}' must be finite."
        ) from None
    if not math.isfinite(number):
        raise LLMClientError(f"Inspection metadata field '{field}' must be finite.")
    return number


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, Any]:
    """Select and validate only the non-identifying metadata sent to the model."""
    required = {
        "decision",
        "score",
        "rule_defect_type",
        "rule_severity",
        "rule_confidence",
    }
    missing = required.difference(metadata)
    if missing:
        raise LLMClientError("Required inspection metadata is missing.")
    if metadata["decision"] != "defect":
        raise LLMClientError(
            "LLM classification is only available for defect decisions."
        )

    rule_defect_type = metadata["rule_defect_type"]
    if (
        not isinstance(rule_defect_type, str)
        or rule_defect_type not in SUPPORTED_DEFECT_TYPES
    ):
        raise LLMClientError("Inspection metadata contains an unsupported defect type.")

    severity = metadata["rule_severity"]
    confidence = metadata["rule_confidence"]
    if not isinstance(severity, str) or severity not in _SEVERITIES:
        raise LLMClientError("Inspection metadata contains an invalid severity.")
    if not isinstance(confidence, str) or confidence not in _SEVERITIES:
        raise LLMClientError("Inspection metadata contains an invalid confidence.")

    safe = {
        "decision": "defect",
        "score": _finite_number(metadata["score"], "score"),
        "rule_defect_type": rule_defect_type,
        "rule_severity": severity,
        "rule_confidence": confidence,
    }
    if "hot_fraction" in metadata:
        hot_fraction = metadata["hot_fraction"]
        if hot_fraction is None:
            safe["hot_fraction"] = None
        else:
            hot_fraction = _finite_number(hot_fraction, "hot_fraction")
            if not 0 <= hot_fraction <= 1:
                raise LLMClientError(
                    "Inspection metadata field 'hot_fraction' must be between 0 and 1."
                )
            safe["hot_fraction"] = hot_fraction
    thresholds = {}
    for field in ("low_threshold", "high_threshold"):
        if field in metadata:
            threshold = _finite_number(metadata[field], field)
            if threshold <= 0:
                raise LLMClientError(
                    f"Inspection metadata field '{field}' must be greater than zero."
                )
            thresholds[field] = threshold
    if thresholds:
        if (
            len(thresholds) != 2
            or thresholds["low_threshold"] >= thresholds["high_threshold"]
        ):
            raise LLMClientError("Inspection metadata contains invalid thresholds.")
        safe.update(thresholds)
    return safe


def _validated_result(value: object, rule_defect_type: str) -> dict[str, str]:
    if not isinstance(value, dict) or frozenset(value) != _RESPONSE_FIELDS:
        raise LLMClientError(
            "Ollama response must contain only defect_type and rationale."
        )

    defect_type = value["defect_type"]
    rationale = value["rationale"]
    if not isinstance(defect_type, str) or defect_type not in SUPPORTED_DEFECT_TYPES:
        raise LLMClientError("Ollama response contains an unsupported defect type.")
    allowed_types = (
        SUPPORTED_DEFECT_TYPES
        if rule_defect_type == "unspecified_anomaly"
        else {rule_defect_type, "unspecified_anomaly"}
    )
    if defect_type not in allowed_types:
        raise LLMClientError(
            "Ollama response contains a defect type unsupported by the rule evidence."
        )
    if (
        not isinstance(rationale, str)
        or not rationale.strip()
        or len(rationale) > 500
    ):
        raise LLMClientError(
            "Ollama response rationale must be a non-empty string of at most 500 "
            "characters."
        )
    return {"defect_type": defect_type, "rationale": rationale.strip()}


def classify_inspection(metadata: Mapping[str, object]) -> dict[str, str]:
    """Ask Ollama to classify a defect using safe, structured metadata only."""
    safe_metadata = _safe_metadata(metadata)
    model, api_url = _client_settings()
    payload = {
        "model": model,
        "stream": False,
        "temperature": 0,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify only the defect subtype for an inspection already marked "
                    "defect. Never change its detector decision, severity, or confidence. "
                    "Do not contradict a specific rule_defect_type. When the rule type is "
                    "unspecified, choose a supported subtype only if the supplied evidence "
                    "supports it. A heatmap location or anomaly score alone does not "
                    "establish a specific defect type. If evidence is insufficient, choose "
                    "unspecified_anomaly. Choose only from: "
                    + ", ".join(sorted(SUPPORTED_DEFECT_TYPES))
                    + '. Return exactly a JSON object with string fields "defect_type" '
                    'and "rationale".'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(safe_metadata, allow_nan=False),
            },
        ],
    }

    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        raise LLMClientError("Ollama request timed out.") from None
    except requests.RequestException as error:
        raise LLMClientError(
            f"Ollama request failed ({type(error).__name__})."
        ) from None

    if not response.ok:
        raise LLMClientError(f"Ollama returned HTTP {response.status_code}.")

    try:
        body = response.json()
    except ValueError:
        raise LLMClientError("Ollama returned an invalid JSON response.") from None

    if not isinstance(body, dict):
        raise LLMClientError("Ollama returned an invalid response structure.")
    message = body.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMClientError("Ollama response did not contain JSON text.")

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        raise LLMClientError("Ollama response was not valid JSON.") from None
    return _validated_result(result, safe_metadata["rule_defect_type"])
