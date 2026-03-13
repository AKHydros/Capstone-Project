from __future__ import annotations

from dataclasses import dataclass


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

    @property
    def value_labels_text(self) -> str:
        return " | ".join(self.value_labels[:20])

    @property
    def topic_labels_text(self) -> str:
        return ", ".join(self.topic_labels)

    @property
    def topic_sources_text(self) -> str:
        pairs = [f"{topic} ({self.topic_label_sources.get(topic, 'Unknown')})" for topic in self.topic_labels]
        return ", ".join(pairs)

    @property
    def document_text(self) -> str:
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
