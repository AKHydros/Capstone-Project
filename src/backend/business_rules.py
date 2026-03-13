from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import QuestionRecord


@dataclass(frozen=True)
class RetrievalRules:
    lexical_weight: float = 0.45
    semantic_weight: float = 0.55
    top_k: int = 12
    min_score_threshold: float = 0.05


@dataclass(frozen=True)
class ChatRules:
    max_cards_display: int = 8
    strict_grounding: bool = True


RETRIEVAL_RULES = RetrievalRules()
CHAT_RULES = ChatRules()

TOPIC_TAXONOMY: dict[str, tuple[str, ...]] = {
    "Demographics": (
        "born",
        "age",
        "gender",
        "province",
        "reside",
        "household",
        "employment status",
        "relationship status",
    ),
    "Financial Planning": (
        "financial plan",
        "retire",
        "retirement",
        "goal",
        "portfolio",
        "invest",
        "advice",
    ),
    "Trust & Sentiment": (
        "trust",
        "confidence",
        "comfort",
        "concern",
        "perception",
        "satisfied",
        "satisfaction",
    ),
    "Digital Behavior": (
        "online",
        "digital",
        "internet",
        "app",
        "mobile",
        "robo",
        "technology",
    ),
    "Product Ownership": (
        "product",
        "account",
        "gic",
        "mutual fund",
        "insurance",
        "mortgage",
        "savings",
    ),
    "Provider Relationship": (
        "financial institution",
        "provider",
        "advisor",
        "company",
        "primary",
        "relationship",
        "bank",
    ),
    "Business Banking": (
        "business",
        "employees",
        "organization",
        "company title",
        "contractor",
        "inception",
    ),
}


def normalize_filter(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def infer_wave_year(variable_name: str) -> str:
    # PMG16_..., PMG22_..., etc.
    if variable_name.startswith("PMG") and len(variable_name) >= 5 and variable_name[3:5].isdigit():
        return f"20{variable_name[3:5]}"
    return "Unknown"


def infer_survey_name(variable_name: str) -> str:
    if variable_name == "Survey_Name":
        return "Survey Metadata"
    parts = variable_name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return "Unknown"


def is_valid_question_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    # Remove clear system fields from question retrieval experience.
    return stripped.lower() not in {"unique identifier", "date started", "date completed"}


def apply_filters(
    records: Iterable[QuestionRecord],
    survey_name: str | None,
    wave_year: str | None,
    topic_label: str | None = None,
    topic_source_type: str | None = None,
) -> list[QuestionRecord]:
    survey_name = normalize_filter(survey_name)
    wave_year = normalize_filter(wave_year)
    topic_label = normalize_filter(topic_label)
    topic_source_type = normalize_filter(topic_source_type)

    out: list[QuestionRecord] = []
    for record in records:
        if survey_name and record.survey_name != survey_name:
            continue
        if wave_year and record.wave_year != wave_year:
            continue
        if topic_label and topic_label not in record.topic_labels:
            continue
        if topic_source_type and topic_source_type != "All":
            if topic_label:
                source = record.topic_label_sources.get(topic_label)
                if source != topic_source_type:
                    continue
            elif topic_source_type not in set(record.topic_label_sources.values()):
                continue
        out.append(record)
    return out


def categorize_question_labels(question_text: str, value_labels: list[str]) -> tuple[list[str], dict[str, str]]:
    question_text_lower = question_text.lower()
    values_text_lower = " ".join(value_labels).lower()
    labels: list[str] = []
    sources: dict[str, str] = {}
    for topic, keywords in TOPIC_TAXONOMY.items():
        q_match = any(keyword in question_text_lower for keyword in keywords)
        v_match = any(keyword in values_text_lower for keyword in keywords)
        if not q_match and not v_match:
            continue
        if q_match and v_match:
            source = "Both"
        elif q_match:
            source = "Question Text"
        else:
            source = "Value Labels"
        if topic not in labels:
            labels.append(topic)
        sources[topic] = source
    if not labels:
        labels.append("General")
        sources["General"] = "Fallback"
    return labels, sources


def build_grounded_context(records: list[QuestionRecord]) -> str:
    lines: list[str] = []
    for idx, record in enumerate(records, start=1):
        lines.append(f"[{idx}] {record.question_id} | {record.question_text}")
        lines.append(
            "     "
            f"survey={record.survey_name}, wave={record.wave_year}, "
            f"topics={record.topic_sources_text}, values={record.value_labels_text}"
        )
    return "\n".join(lines)
