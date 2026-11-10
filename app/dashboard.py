"""Streamlit dashboard for inspection, ticket, review, and report workflows."""

import os
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from agent.db import InMemoryDB, SupabaseDB, SupabaseError
from agent.tools import classify_defect, create_ticket, fetch_procedure, log_decision
from app.dashboard_data import (
    build_report_html,
    image_markup,
    ordered_logs,
    read_thresholds,
    resolve_heatmap_path,
    resolve_image_path,
    seed_offline_db,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE = os.getenv("DASHBOARD_OFFLINE", "").strip() == "1"
DECISIONS = ("normal", "uncertain", "defect")
TICKET_STATUSES = ("open", "in_progress", "closed")

st.set_page_config(page_title="InspectAI Quality Dashboard", page_icon="🔎", layout="wide")


def _show_error(action: str, error: Exception) -> None:
    """Show a short safe message instead of a traceback or credential."""
    st.error(
        f"{action} failed. Check your connection and Supabase access. "
        f"Error type: {type(error).__name__}."
    )


def _database():
    """Reuse one database per browser session so offline edits persist on reruns."""
    if "dashboard_db" not in st.session_state:
        if OFFLINE:
            st.session_state.dashboard_db = seed_offline_db()
        else:
            st.session_state.dashboard_db = SupabaseDB()
    return st.session_state.dashboard_db


def _threshold_values() -> dict[str, float]:
    return read_thresholds()


def _render_table(headers: list[str], rows: list[list[Any]]) -> None:
    """Render a small HTML table without pandas or Streamlit dataframe widgets."""
    header_html = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    if not rows:
        body_html = f'<tr><td colspan="{len(headers)}">No records found.</td></tr>'
    components.html(
        f"""<style>table{{border-collapse:collapse;width:100%;font:14px Arial}}
        th,td{{border:1px solid #bbb;padding:8px;text-align:left;overflow-wrap:anywhere}}
        th{{background:#edf2f7}}</style><table><thead><tr>{header_html}</tr></thead>
        <tbody>{body_html}</tbody></table>""",
        height=min(600, 70 + 42 * max(len(rows), 1)),
        scrolling=True,
    )


def _records_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def _inspection_context(db: Any, inspection_id: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inspections = db.list_inspections(inspection_id=inspection_id)
    if not inspections:
        raise ValueError("The related inspection could not be found.")
    return inspections[0], db.list_log(inspection_id=inspection_id)


def _pending_reviews(db: Any) -> list[dict[str, Any]]:
    return [
        review for review in db.list_reviews()
        if not str(review.get("reviewer_label") or "").strip()
    ]


def _overview(db: Any) -> None:
    st.title("Overview")
    inspections = db.list_inspections()
    tickets = db.list_tickets()
    pending = _pending_reviews(db)
    by_decision = Counter(str(row.get("decision", "")).lower() for row in inspections)
    total = len(inspections)
    review_ids = {str(row.get("inspection_id")) for row in db.list_reviews()}
    share = 100 * len(review_ids) / total if total else 0
    open_tickets = sum(row.get("status") == "open" for row in tickets)

    metrics = [
        ("Total inspections", total),
        ("Normal", by_decision["normal"]),
        ("Uncertain", by_decision["uncertain"]),
        ("Defect", by_decision["defect"]),
        ("Open tickets", open_tickets),
        ("Pending reviews", len(pending)),
        ("Cases sent to review", f"{share:.1f}%"),
    ]
    columns = st.columns(4)
    for index, (label, value) in enumerate(metrics):
        columns[index % 4].metric(label, value)

    st.subheader("Detector decisions")
    colors = {"normal": "#3b9b66", "uncertain": "#dda72f", "defect": "#d65252"}
    max_count = max((by_decision[item] for item in DECISIONS), default=0) or 1
    bars = "".join(
        f"""<div style="display:flex;align-items:center;gap:10px;margin:10px 0">
        <span style="width:90px">{escape(label.title())}</span>
        <div style="background:{colors[label]};height:22px;width:{by_decision[label] / max_count * 60:.1f}%;min-width:2px"></div>
        <strong>{by_decision[label]}</strong></div>"""
        for label in DECISIONS
    )
    components.html(bars, height=150)
    try:
        thresholds = _threshold_values()
        st.subheader("Detector thresholds")
        st.write(f"Low threshold: `{thresholds['low_threshold']:.6f}`")
        st.write(f"High threshold: `{thresholds['high_threshold']:.6f}`")
    except (OSError, ValueError, KeyError) as error:
        st.warning(f"Thresholds are unavailable: {error}")


def _inspections_page(db: Any) -> None:
    st.title("Inspections")
    decision = st.selectbox("Filter by detector decision", ["all", *DECISIONS])
    inspections = db.list_inspections(status=None if decision == "all" else decision)
    _render_table(
        ["ID", "Image path", "Score", "Decision", "Created"],
        [
            [
                row.get("id", ""),
                row.get("image_path", ""),
                row.get("score", ""),
                row.get("decision", ""),
                row.get("created_at", ""),
            ]
            for row in inspections
        ],
    )
    if not inspections:
        return

    options = _records_by_id(inspections)
    selected_id = st.selectbox(
        "Select an inspection",
        list(options),
        format_func=lambda key: (
            f"#{key} — {options[key].get('image_path', '')}"
        ),
    )
    inspection = options[selected_id]
    original_path = resolve_image_path(str(inspection.get("image_path", "")))
    heatmap_path = resolve_heatmap_path(str(inspection.get("heatmap_path", "")))
    left, right = st.columns(2)
    for column, path, label in (
        (left, original_path, "Original image"),
        (right, heatmap_path, "Heatmap"),
    ):
        column.subheader(label)
        components.html(
            image_markup(path, label),
            height=440,
            scrolling=False,
        )

    try:
        _threshold_values()
        score = float(inspection.get("score", 0))
        st.subheader("Score and thresholds")
        st.write(
            f"Score `{score:.6f}` · low `{thresholds['low_threshold']:.6f}` · "
            f"high `{thresholds['high_threshold']:.6f}`"
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.warning(f"Could not compare score with thresholds: {error}")

    inspection_id = inspection.get("id")
    tickets = db.list_tickets(inspection_id=inspection_id)
    st.subheader("Ticket")
    if tickets:
        _render_table(
            ["Ticket ID", "Type", "Severity", "Status"],
            [[row.get("id", ""), row.get("defect_type", ""), row.get("severity", ""), row.get("status", "")] for row in tickets],
        )
    else:
        st.info("No ticket is linked to this inspection.")

    st.subheader("Agent decision log")
    logs = ordered_logs(db.list_log(inspection_id=inspection_id))
    _render_table(
        ["Time", "Step", "Tool", "Input", "Output"],
        [[row.get("created_at", ""), row.get("step", ""), row.get("tool_name", ""), row.get("input_summary", ""), row.get("output_summary", "")] for row in logs],
    )


def _tickets_page(db: Any) -> None:
    st.title("Tickets")
    tickets = db.list_tickets()
    status_filter = st.selectbox("Status", ["all", *TICKET_STATUSES])
    severities = sorted({str(row.get("severity", "")) for row in tickets if row.get("severity")})
    severity_filter = st.selectbox("Severity", ["all", *severities])
    defect_types = sorted({str(row.get("defect_type", "")) for row in tickets if row.get("defect_type")})
    type_filter = st.selectbox("Defect type", ["all", *defect_types])
    filtered = [
        row for row in tickets
        if (status_filter == "all" or row.get("status") == status_filter)
        and (severity_filter == "all" or row.get("severity") == severity_filter)
        and (type_filter == "all" or row.get("defect_type") == type_filter)
    ]
    _render_table(
        ["ID", "Inspection", "Defect type", "Severity", "Status", "Created"],
        [[row.get("id", ""), row.get("inspection_id", ""), row.get("defect_type", ""), row.get("severity", ""), row.get("status", ""), row.get("created_at", "")] for row in filtered],
    )
    if not filtered:
        return

    ticket_options = _records_by_id(filtered)
    selected = st.selectbox("Select a ticket to update", list(ticket_options))
    ticket = ticket_options[selected]
    new_status = st.selectbox(
        "New status",
        TICKET_STATUSES,
        index=TICKET_STATUSES.index(ticket.get("status", "open")),
        key=f"ticket_status_{selected}",
    )
    if st.button("Save ticket status", key=f"save_ticket_{selected}"):
        try:
            db.update_ticket_status(ticket["id"], new_status)
            st.success("Ticket status saved.")
            st.rerun()
        except Exception as error:
            _show_error("Saving ticket status", error)


def _review_queue_page(db: Any) -> None:
    st.title("Review queue")
    pending = _pending_reviews(db)
    if not pending:
        st.info("There are no pending reviews.")
        return

    for review in pending:
        inspection_id = review.get("inspection_id")
        st.divider()
        st.subheader(f"Review #{review.get('id')} · inspection #{inspection_id}")
        st.write(review.get("reason", "No reason recorded."))
        try:
            inspection, _ = _inspection_context(db, inspection_id)
        except Exception as error:
            _show_error("Loading review inspection", error)
            continue
        st.write(f"Score: `{float(inspection.get('score', 0)):.6f}`")
        first, second = st.columns(2)
        with first:
            components.html(
                image_markup(resolve_image_path(str(inspection.get("image_path", ""))), "Original"),
                height=440,
                scrolling=False,
            )
        with second:
            components.html(
                image_markup(resolve_heatmap_path(str(inspection.get("heatmap_path", ""))), "Heatmap"),
                height=440,
                scrolling=False,
            )
        with st.form(f"review_form_{review.get('id')}"):
            notes = st.text_area("Review notes", key=f"review_notes_{review.get('id')}")
            defect = st.form_submit_button("Confirm defect")
            normal = st.form_submit_button("Confirm normal")
        if defect or normal:
            reviewer_label = "confirmed_defect" if defect else "confirmed_normal"
            try:
                # Confirming a defect uses the agent's own classification and ticket tools.
                if defect:
                    classification = classify_defect(
                        db,
                        float(inspection["score"]),
                        resolve_heatmap_path(str(inspection.get("heatmap_path", ""))),
                    )
                    procedure = fetch_procedure(db, classification["defect_type"])
                    create_ticket(
                        db,
                        inspection_id,
                        classification["defect_type"],
                        classification["severity"],
                        procedure,
                    )
                db.resolve_review(review["id"], reviewer_label, notes)
                log_decision(
                    db,
                    inspection_id,
                    step="manual_review",
                    tool_name="reviewer",
                    input_summary=f"review_id={review['id']}",
                    output_summary=f"{reviewer_label}: {notes}",
                )
                st.success("Review decision saved.")
                st.rerun()
            except Exception as error:
                _show_error("Saving review decision", error)


def _reports_page(db: Any) -> None:
    st.title("Reports")
    tickets = db.list_tickets()
    if not tickets:
        st.info("No tickets are available for reporting.")
        return
    ticket_options = _records_by_id(tickets)
    selected = st.selectbox(
        "Choose a ticket",
        list(ticket_options),
        format_func=lambda key: (
            f"Ticket #{key} · inspection #{ticket_options[key].get('inspection_id')} · "
            f"{ticket_options[key].get('defect_type', '')}"
        ),
    )
    ticket = ticket_options[selected]
    try:
        inspection, _ = _inspection_context(db, ticket.get("inspection_id"))
        logs = ordered_logs(db.list_log(inspection_id=ticket.get("inspection_id")))
        report = build_report_html(inspection, ticket, logs)
    except Exception as error:
        _show_error("Building report", error)
        return

    st.caption(f"Report traces to inspection ID {inspection.get('id')}.")
    components.html(report, height=900, scrolling=True)
    st.download_button(
        "Download standalone HTML report",
        data=report,
        file_name=f"inspection-{inspection.get('id')}-report.html",
        mime="text/html",
    )


def main() -> None:
    st.sidebar.title("InspectAI")
    if OFFLINE:
        st.sidebar.warning("OFFLINE MODE — using in-memory demo data")
        st.markdown(
            '<div style="background:#fff3cd;padding:10px;border:1px solid #e0b84c">'
            "OFFLINE MODE — changes are temporary and are not sent to Supabase."
            "</div>",
            unsafe_allow_html=True,
        )
    pages = {
        "Overview": _overview,
        "Inspections": _inspections_page,
        "Tickets": _tickets_page,
        "Review queue": _review_queue_page,
        "Reports": _reports_page,
    }
    selected_page = st.sidebar.radio("Page", list(pages))
    try:
        db = _database()
    except Exception as error:
        _show_error("Connecting to the inspection database", error)
        return

    try:
        pages[selected_page](db)
    except SupabaseError as error:
        _show_error("Loading dashboard data", error)
    except Exception as error:
        _show_error("Loading dashboard page", error)


if __name__ == "__main__":
    main()
