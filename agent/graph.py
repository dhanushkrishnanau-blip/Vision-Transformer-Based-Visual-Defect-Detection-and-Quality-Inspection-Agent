"""LangGraph workflow that routes detections into logs, reviews, or tickets."""

from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agent import tools
from agent.tools import (
    classify_defect,
    create_ticket,
    fetch_procedure,
    log_decision,
    request_review,
)


class AgentState(TypedDict, total=False):
    """Values passed between the graph's small processing steps."""

    db: Any
    detection: dict[str, str]
    inspection_id: Any
    decision: str
    score: float
    image_path: str
    heatmap_path: str
    heatmap_file: str
    defect_type: str
    severity: str
    confidence: str
    procedure_text: str
    stage: str
    error: str
    review_attempted: bool
    ticket_created: bool
    review_created: bool
    log_error: str


def _error_text(error: Exception) -> str:
    """Keep useful error context while ensuring accidental secrets stay hidden."""
    message = str(error).replace("\n", " ")
    db = getattr(error, "db", None)
    secret = getattr(db, "secret_key", "")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return f"{type(error).__name__}: {message[:300]}"


def _load_detection(state: AgentState) -> dict[str, Any]:
    """Upsert one CSV row and remember its database ID."""
    detection = state["detection"]
    row = {
        "image_path": detection["image_path"],
        "heatmap_path": detection["heatmap_path"],
        "score": float(detection["score"]),
        "decision": detection["decision"],
    }
    inspection = state["db"].upsert_inspection(row)
    if "id" not in inspection:
        raise ValueError("The database did not return an inspection ID.")
    return {
        "inspection_id": inspection["id"],
        "decision": row["decision"],
        "score": row["score"],
        "image_path": row["image_path"],
        "heatmap_path": row["heatmap_path"],
        "stage": "load_detection",
        "error": "",
    }


def _route(state: AgentState) -> Literal[
    "classify_defect", "fetch_procedure", "create_ticket",
    "request_review", "log_decision", "handle_error", "end",
]:
    """Route by the detector's decision column and the current processing step."""
    if state.get("error"):
        return "handle_error"
    stage = state.get("stage")
    if stage == "load_detection":
        decision = state.get("decision")
        if decision == "normal":
            return "log_decision"
        if decision == "uncertain":
            return "request_review"
        if decision == "defect":
            return "classify_defect"
        raise ValueError(f"Unknown detector decision: {decision}")
    if stage == "classify_defect":
        return (
            "request_review"
            if state.get("confidence") == "low"
            else "fetch_procedure"
        )
    if stage == "fetch_procedure":
        return "create_ticket"
    if stage in {"create_ticket", "request_review"}:
        return "log_decision"
    if stage == "log_decision":
        return "end"
    return "handle_error"


def _classify(state: AgentState) -> dict[str, Any]:
    result = classify_defect(
        state["db"],
        state["score"],
        state.get("heatmap_file"),
    )
    return {**result, "stage": "classify_defect"}


def _get_procedure(state: AgentState) -> dict[str, Any]:
    return {
        "procedure_text": fetch_procedure(state["db"], state["defect_type"]),
        "stage": "fetch_procedure",
    }


def _create_ticket(state: AgentState) -> dict[str, Any]:
    _, created = create_ticket(
        state["db"],
        state["inspection_id"],
        state["defect_type"],
        state["severity"],
        state["procedure_text"],
    )
    return {"ticket_created": created, "stage": "create_ticket"}


def _request_review(state: AgentState) -> dict[str, Any]:
    if state.get("confidence") == "low":
        reason = "Classification confidence is low."
    else:
        reason = f"Detector decision: {state.get('decision')}"
    _, created = request_review(state["db"], state["inspection_id"], reason)
    return {
        "review_created": created,
        "review_attempted": True,
        "stage": "request_review",
    }


def _log(state: AgentState) -> dict[str, Any]:
    result = log_decision(
        state["db"],
        state["inspection_id"],
        step=state.get("stage", "unknown"),
        tool_name="inspection_agent",
        input_summary=f"decision={state.get('decision')}; score={state.get('score')}",
        output_summary=(
            f"defect_type={state.get('defect_type', '')}; "
            f"ticket_created={state.get('ticket_created', False)}; "
            f"review_created={state.get('review_created', False)}"
        ),
    )
    return {"stage": "log_decision", "logged": bool(result)}


def _handle_error(state: AgentState) -> dict[str, Any]:
    """Best-effort error log and review request; never loops on repeated errors."""
    message = state.get("error", "Unknown tool error")
    inspection_id = state.get("inspection_id")
    db = state["db"]
    log_error = ""
    try:
        if inspection_id is not None:
            log_decision(
                db,
                inspection_id,
                step=state.get("stage", "unknown"),
                tool_name="agent_error",
                input_summary=f"decision={state.get('decision')}",
                output_summary=message,
            )
    except Exception as error:
        log_error = _error_text(error)

    review_created = False
    review_error = ""
    if inspection_id is not None:
        try:
            _, review_created = request_review(
                db, inspection_id, f"Agent tool error: {message}"
            )
        except Exception as error:
            review_error = _error_text(error)

    return {
        "error": message,
        "log_error": log_error,
        "review_error": review_error,
        "review_created": review_created,
        "review_attempted": True,
        "stage": "error_handled",
    }


def _safe(function):
    """Catch node errors so the graph can divert the case to human review."""
    def run(state: AgentState) -> dict[str, Any]:
        try:
            return function(state)
        except Exception as error:
            return {"error": _error_text(error), "stage": function.__name__}

    return run


def build_graph():
    """Create the inspection state graph."""
    graph = StateGraph(AgentState)
    graph.add_node("load_detection", _safe(_load_detection))
    graph.add_node("route", lambda state: state)
    graph.add_node("classify_defect", _safe(_classify))
    graph.add_node("fetch_procedure", _safe(_get_procedure))
    graph.add_node("create_ticket", _safe(_create_ticket))
    graph.add_node("request_review", _safe(_request_review))
    graph.add_node("log_decision", _safe(_log))
    graph.add_node("handle_error", _safe(_handle_error))

    graph.set_entry_point("load_detection")
    graph.add_edge("load_detection", "route")
    for node in (
        "classify_defect",
        "fetch_procedure",
        "create_ticket",
        "request_review",
        "log_decision",
    ):
        graph.add_edge(node, "route")
    graph.add_conditional_edges(
        "route",
        _route,
        {
            "classify_defect": "classify_defect",
            "fetch_procedure": "fetch_procedure",
            "create_ticket": "create_ticket",
            "request_review": "request_review",
            "log_decision": "log_decision",
            "handle_error": "handle_error",
            "end": END,
        },
    )
    graph.add_edge("handle_error", END)
    return graph.compile()


_GRAPH = build_graph()


def run_detection(db: Any, detection: dict[str, str]) -> AgentState:
    """Run one detector CSV row through the workflow."""
    image_path = detection["image_path"]
    heatmap_path = detection["heatmap_path"]
    local_heatmap = Path(__file__).resolve().parents[1] / "demo_data" / "heatmaps"
    state: AgentState = {
        "db": db,
        "detection": detection,
        "heatmap_file": str(local_heatmap / Path(heatmap_path)),
    }
    return _GRAPH.invoke(state)
