from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..models import QuestionRecord
from .governance_service import GovernanceService


@dataclass(frozen=True)
class QuestionLibrary:
    top_questions: list[str]
    generated_questions: list[str]


class QuestionLibraryService:
    def __init__(self, governance_service: GovernanceService) -> None:
        """Stores governance dependency for preloaded question persistence."""
        self.governance_service = governance_service

    def preload(self, records: list[QuestionRecord], iterations: int = 10) -> QuestionLibrary:
        """Builds top/generated question lists and persists them into governance store."""
        top_questions = self._derive_top_questions(records, limit=50)
        generated = self._generate_questions(records, iterations=iterations)

        for text in top_questions:
            self.governance_service.save_qa_item(
                question_text=text,
                source="dataset_heuristic",
                status="Approved",
                labels=["Common"],
                takeaways=["Preloaded from dataset frequency heuristics."],
                last_trace_id=None,
            )

        for text in generated:
            self.governance_service.save_qa_item(
                question_text=text,
                source="dynamic_generation",
                status="Draft",
                labels=["Generated"],
                takeaways=["Generated from topic trends."],
                last_trace_id=None,
            )

        return QuestionLibrary(top_questions=top_questions, generated_questions=generated)

    def _derive_top_questions(self, records: list[QuestionRecord], limit: int) -> list[str]:
        """Produces deterministic top-question list from ordered records."""
        ordered = sorted(records, key=lambda r: (r.survey_name, r.wave_year, r.question_id))
        return [record.question_text for record in ordered[:limit]]

    def _generate_questions(self, records: list[QuestionRecord], iterations: int) -> list[str]:
        """Generates heuristic question prompts from dominant topic labels."""
        topic_counts = Counter(topic for record in records for topic in record.topic_labels)
        common_topics = [topic for topic, _ in topic_counts.most_common(max(iterations, 1))]

        generated: list[str] = []
        for idx in range(iterations):
            topic = common_topics[idx % len(common_topics)] if common_topics else "General"
            template = (
                f"What are the strongest trends for {topic.lower()} across surveys and waves?"
                if idx % 2 == 0
                else f"Which variables best explain {topic.lower()} differences by wave/year?"
            )
            generated.append(template)
        return generated
