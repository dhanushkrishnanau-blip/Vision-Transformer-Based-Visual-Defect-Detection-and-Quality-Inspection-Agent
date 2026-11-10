"""Small Supabase REST client and in-memory database for the inspection agent."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv

TABLES = {
    "inspections": "inspections",
    "tickets": "tickets",
    "reviews": "reviews",
    "log": "decision_log",
}


class SupabaseError(RuntimeError):
    """Raised when a Supabase request fails."""


class SupabaseDB:
    """Access the existing Supabase tables through their REST API."""

    def __init__(self) -> None:
        # Load secrets for use in memory; never print or save them.
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        self.url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        self.secret_key = os.getenv("SUPABASE_SECRET_KEY", "").strip()
        if not self.url:
            raise SupabaseError("SUPABASE_URL is missing from .env.")
        if not self.secret_key:
            raise SupabaseError("SUPABASE_SECRET_KEY is missing from .env.")

    def _request(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        prefer: str | None = None,
        count_only: bool = False,
    ) -> tuple[Any, str]:
        """Send one REST request and return its JSON body and count header."""
        headers = {"apikey": self.secret_key}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        if count_only:
            headers["Range-Unit"] = "items"
            headers["Range"] = "0-0"

        try:
            response = requests.request(
                method,
                f"{self.url}/rest/v1/{table}",
                params=params,
                json=payload,
                headers=headers,
                timeout=30,
            )
        except requests.RequestException as error:
            raise SupabaseError(
                f"Supabase {method} request to {table} failed ({type(error).__name__})."
            ) from None

        # Remove the secret if a server ever echoes request details in an error.
        response_text = response.text.replace(self.secret_key, "[REDACTED]")
        if not response.ok:
            raise SupabaseError(
                f"Supabase {method} {table} returned HTTP {response.status_code}: "
                f"{response_text[:500]}"
            )
        if not response_text:
            return [], response.headers.get("Content-Range", "")
        try:
            return response.json(), response.headers.get("Content-Range", "")
        except requests.exceptions.JSONDecodeError:
            return response_text, response.headers.get("Content-Range", "")

    def upsert_inspection(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert or update an inspection using its unique image path."""
        body, _ = self._request(
            "POST",
            TABLES["inspections"],
            params={"on_conflict": "image_path"},
            payload=row,
            prefer="resolution=merge-duplicates,return=representation",
        )
        if not body:
            raise SupabaseError("Supabase returned no inspection row after upsert.")
        return body[0] if isinstance(body, list) else body

    def create_ticket(self, row: dict[str, Any]) -> dict[str, Any]:
        body, _ = self._request(
            "POST", TABLES["tickets"], payload=row, prefer="return=representation"
        )
        return body[0] if isinstance(body, list) and body else body

    def add_review(self, row: dict[str, Any]) -> dict[str, Any]:
        body, _ = self._request(
            "POST", TABLES["reviews"], payload=row, prefer="return=representation"
        )
        return body[0] if isinstance(body, list) and body else body

    def update_ticket_status(self, ticket_id: Any, status: str) -> dict[str, Any]:
        if status not in {"open", "in_progress", "closed"}:
            raise ValueError(f"Invalid ticket status: {status}")
        body, _ = self._request(
            "PATCH",
            TABLES["tickets"],
            params={"id": f"eq.{ticket_id}"},
            payload={"status": status},
            prefer="return=representation",
        )
        if not body:
            raise SupabaseError(f"Ticket {ticket_id} was not found.")
        return body[0] if isinstance(body, list) else body

    def resolve_review(
        self, review_id: Any, reviewer_label: str, notes: str = ""
    ) -> dict[str, Any]:
        body, _ = self._request(
            "PATCH",
            TABLES["reviews"],
            params={"id": f"eq.{review_id}"},
            payload={
                "reviewer_label": reviewer_label,
                "notes": notes,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            },
            prefer="return=representation",
        )
        if not body:
            raise SupabaseError(f"Review {review_id} was not found.")
        return body[0] if isinstance(body, list) else body

    def log(self, row: dict[str, Any]) -> dict[str, Any]:
        body, _ = self._request(
            "POST", TABLES["log"], payload=row, prefer="return=representation"
        )
        return body[0] if isinstance(body, list) and body else body

    def list_inspections(
        self, inspection_id: Any = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        params = {"id": f"eq.{inspection_id}"} if inspection_id is not None else {}
        if status is not None:
            params["decision"] = f"eq.{status}"
        body, _ = self._request("GET", TABLES["inspections"], params=params)
        return body if isinstance(body, list) else []

    def list_tickets(
        self, inspection_id: Any = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        params = {}
        if inspection_id is not None:
            params["inspection_id"] = f"eq.{inspection_id}"
        if status is not None:
            params["status"] = f"eq.{status}"
        body, _ = self._request("GET", TABLES["tickets"], params=params)
        return body if isinstance(body, list) else []

    def list_reviews(self, inspection_id: Any = None) -> list[dict[str, Any]]:
        params = (
            {"inspection_id": f"eq.{inspection_id}"}
            if inspection_id is not None
            else {}
        )
        body, _ = self._request("GET", TABLES["reviews"], params=params)
        return body if isinstance(body, list) else []

    def list_log(self, inspection_id: Any = None) -> list[dict[str, Any]]:
        params = (
            {"inspection_id": f"eq.{inspection_id}"}
            if inspection_id is not None
            else {}
        )
        body, _ = self._request("GET", TABLES["log"], params=params)
        return body if isinstance(body, list) else []

    def count(self, table_name: str) -> int:
        """Read only a table's exact row count without returning row contents."""
        if table_name not in TABLES:
            raise ValueError(f"Unknown table: {table_name}")
        _, content_range = self._request(
            "GET",
            TABLES[table_name],
            params={"select": "id"},
            prefer="count=exact",
            count_only=True,
        )
        try:
            return int(content_range.rsplit("/", 1)[1])
        except (IndexError, ValueError) as error:
            raise SupabaseError(
                f"Supabase did not return an exact count for {table_name}."
            ) from error


class InMemoryDB:
    """Store rows in lists so tests and offline runs need no network."""

    def __init__(self) -> None:
        self.inspections: list[dict[str, Any]] = []
        self.tickets: list[dict[str, Any]] = []
        self.reviews: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []
        self._next_id = 1

    def _new_row(self, row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result.setdefault("id", self._next_id)
        self._next_id += 1
        result.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        return result

    def upsert_inspection(self, row: dict[str, Any]) -> dict[str, Any]:
        for existing in self.inspections:
            if existing["image_path"] == row["image_path"]:
                existing.update(row)
                return dict(existing)
        created = self._new_row(row)
        self.inspections.append(created)
        return dict(created)

    def create_ticket(self, row: dict[str, Any]) -> dict[str, Any]:
        created = self._new_row(row)
        self.tickets.append(created)
        return dict(created)

    def add_review(self, row: dict[str, Any]) -> dict[str, Any]:
        created = self._new_row(row)
        self.reviews.append(created)
        return dict(created)

    def update_ticket_status(self, ticket_id: Any, status: str) -> dict[str, Any]:
        if status not in {"open", "in_progress", "closed"}:
            raise ValueError(f"Invalid ticket status: {status}")
        for ticket in self.tickets:
            if ticket["id"] == ticket_id:
                ticket["status"] = status
                return dict(ticket)
        raise ValueError(f"Ticket {ticket_id} was not found.")

    def resolve_review(
        self, review_id: Any, reviewer_label: str, notes: str = ""
    ) -> dict[str, Any]:
        for review in self.reviews:
            if review["id"] == review_id:
                review.update(
                    reviewer_label=reviewer_label,
                    notes=notes,
                    reviewed_at=datetime.now(timezone.utc).isoformat(),
                )
                return dict(review)
        raise ValueError(f"Review {review_id} was not found.")

    def log(self, row: dict[str, Any]) -> dict[str, Any]:
        created = self._new_row(row)
        self.logs.append(created)
        return dict(created)

    def list_inspections(
        self, inspection_id: Any = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.inspections
            if (inspection_id is None or row.get("id") == inspection_id)
            and (status is None or row.get("decision") == status)
        ]

    def list_tickets(
        self, inspection_id: Any = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.tickets
            if (inspection_id is None or row.get("inspection_id") == inspection_id)
            and (status is None or row.get("status") == status)
        ]

    def list_reviews(self, inspection_id: Any = None) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.reviews
            if inspection_id is None or row.get("inspection_id") == inspection_id
        ]

    def list_log(self, inspection_id: Any = None) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.logs
            if inspection_id is None or row.get("inspection_id") == inspection_id
        ]

    def count(self, table_name: str) -> int:
        if table_name not in TABLES:
            raise ValueError(f"Unknown table: {table_name}")
        attribute = "logs" if table_name == "log" else table_name
        return len(getattr(self, attribute))
