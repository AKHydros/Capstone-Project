from __future__ import annotations

from dataclasses import dataclass

from ..models import QuestionRecord
from ..observability.db import ObservabilityStore


VALID_STATUSES = {"Draft", "Approved", "Deprecated"}


@dataclass(frozen=True)
class GovernanceLabels:
    labels: list[str]
    takeaways: list[str]


class GovernanceService:
    def __init__(self, store: ObservabilityStore) -> None:
        """Stores observability/governance persistence dependency."""
        self.store = store

    def generate_labels_takeaways(
        self,
        *,
        query: str,
        answer: str,
        ranked_results: list[QuestionRecord],
    ) -> GovernanceLabels:
        """Derives governance labels and key takeaways from ranked results."""
        labels = sorted({label for r in ranked_results for label in r.topic_labels})[:5]
        if not labels:
            labels = ["General"]

        top_items = ranked_results[:3]
        takeaways: list[str] = []
        for record in top_items:
            takeaway = f"{record.question_id}: {record.question_text[:80]}"
            takeaways.append(takeaway)
        if not takeaways:
            takeaways = [answer[:120]]

        return GovernanceLabels(labels=labels, takeaways=takeaways)

    def save_qa_item(
        self,
        *,
        question_text: str,
        source: str,
        status: str,
        labels: list[str],
        takeaways: list[str],
        last_trace_id: str | None,
    ) -> int:
        """Saves or updates governance item with validated status."""
        normalized_status = status if status in VALID_STATUSES else "Draft"
        return self.store.upsert_governance_item(
            question_text=question_text,
            source=source,
            status=normalized_status,
            labels=labels,
            takeaways=takeaways,
            last_trace_id=last_trace_id,
        )

    def set_status(self, item_id: int, status: str) -> None:
        """Validates and updates governance item status."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid governance status: {status}")
        self.store.update_governance_status(item_id, status)

    def approved_library(self, limit: int = 100) -> list[dict[str, object]]:
        """Returns approved governance library items."""
        return self.store.list_governance_items(status="Approved", limit=limit)
