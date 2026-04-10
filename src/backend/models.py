from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    question_text: str
    measurement_level: str
    role: str
    source_file: str
    survey_name: str
    wave_year: str
    value_labels: list[str]
    topic_labels: list[str]
    topic_label_sources: dict[str, str]
    context_fields: dict[str, str] = field(default_factory=dict)

    @property
    def value_labels_text(self) -> str:
        """Returns compact joined string of value labels (up to 20)."""
        return " | ".join(self.value_labels[:20])

    @property
    def topic_labels_text(self) -> str:
        """Returns comma-separated topic labels."""
        return ", ".join(self.topic_labels)

    @property
    def topic_sources_text(self) -> str:
        """Returns topic labels annotated with source attribution."""
        pairs = [f"{topic} ({self.topic_label_sources.get(topic, 'Unknown')})" for topic in self.topic_labels]
        return ", ".join(pairs)

    @property
    def document_text(self) -> str:
        """Returns normalized text payload used for indexing/retrieval."""
        return "\n".join(
            [
                f"Question ID: {self.question_id}",
                f"Question Text: {self.question_text}",
                f"Survey: {self.survey_name}",
                f"Wave/Year: {self.wave_year}",
                f"Measurement Level: {self.measurement_level}",
                f"Role: {self.role}",
                f"Topic Labels: {self.topic_sources_text}",
                f"Value Labels: {self.value_labels_text}",
            ]
        )


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    ranked_results: list[QuestionRecord]
    lookup_mode: str = "hybrid_fallback"
    variant_count: int = 0
    retrieval_mode: str = "deterministic"
    answer_mode: str = "direct_answer"
    needs_clarification: bool = False
    confidence_score: float | None = None
    embedding_cache_hit: bool = False
    answer_cache_hit: bool = False
    follow_up_suggestion: str | None = None
    suggestions: list[str] = field(default_factory=list)
