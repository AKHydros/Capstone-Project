from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import QuestionRecord


@dataclass(frozen=True)
class RetrievalRules:
    lexical_weight: float = 0.45
    semantic_weight: float = 0.55
    top_k: int = 20
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
    """Normalizes optional filter values by trimming and handling empty input."""
    if not value:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def infer_wave_year(variable_name: str) -> str:
    """Derive the four-digit survey year from a PMG variable ID.

    PMG variable IDs follow the convention ``PMG{YY}_{SURVEY}_q{N}`` where
    ``{YY}`` is the two-digit year suffix (e.g. ``16`` → ``2016``).

    Parameters
    ----------
    variable_name:
        Raw variable ID string, e.g. ``"PMG22_WAI_q5a"``.

    Returns
    -------
    str
        Four-digit year string, e.g. ``"2022"``, or ``"Unknown"`` when the
        naming convention cannot be matched.
    """
    if variable_name.startswith("PMG") and len(variable_name) >= 5 and variable_name[3:5].isdigit():
        return f"20{variable_name[3:5]}"
    return "Unknown"


def infer_survey_name(variable_name: str) -> str:
    """Derive the survey key prefix from a variable ID.

    For ``PMG22_WAI_q5a`` this returns ``"PMG22_WAI"``.

    Parameters
    ----------
    variable_name:
        Raw variable ID string.

    Returns
    -------
    str
        The first two underscore-delimited segments joined by ``_``, or
        ``"Unknown"`` for variables with fewer than two segments.
        ``"Survey_Name"`` is treated as a special sentinel and mapped to
        ``"Survey Metadata"``.
    """
    if variable_name == "Survey_Name":
        return "Survey Metadata"
    parts = variable_name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return "Unknown"


def is_valid_question_text(text: str) -> bool:
    """Return ``True`` when *text* is a usable question label.

    Filters out empty strings and known system-field labels (e.g.
    ``"Unique Identifier"``, ``"Date Started"``) that appear in the Excel
    source but should never be surfaced in the chatbot.

    Parameters
    ----------
    text:
        Raw label string from the Excel ``Variable Information`` sheet.

    Returns
    -------
    bool
    """
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.lower() not in {"unique identifier", "date started", "date completed"}


def apply_filters(
    records: Iterable[QuestionRecord],
    survey_name: str | None,
    wave_year: str | None,
    topic_label: str | None = None,
    topic_source_type: str | None = None,
) -> list[QuestionRecord]:
    """Apply deterministic equality filters over a stream of records.

    All filter values are normalised (stripped, ``None``-ified if empty)
    before comparison.  A ``None`` filter value means "no constraint" and is
    skipped entirely.

    Parameters
    ----------
    records:
        Iterable of :class:`~backend.models.QuestionRecord` to filter.
    survey_name:
        If set, only records whose ``survey_name`` matches exactly are kept.
    wave_year:
        If set, only records whose ``wave_year`` matches exactly are kept.
    topic_label:
        If set, only records whose ``topic_labels`` list contains this value
        are kept.
    topic_source_type:
        If set alongside *topic_label*, filters by the source attribution for
        that specific topic.  If set without *topic_label*, filters records
        where any topic has this source type.

    Returns
    -------
    list[QuestionRecord]
        Filtered records in original iteration order.
    """
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
    """Map a question and its value labels to taxonomy topics and source metadata.

    Iterates ``TOPIC_TAXONOMY`` and checks whether each topic's keywords appear
    in the question text, in the combined value-label text, or in both.  The
    ``source`` attribute records which text source triggered the match
    (``"Question Text"``, ``"Value Labels"``, or ``"Both"``).

    If no taxonomy topic matches, the record is assigned the fallback topic
    ``"General"`` with source ``"Fallback"``.

    Parameters
    ----------
    question_text:
        The human-readable question label from the Excel source.
    value_labels:
        List of coded value-label strings for this question.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        ``(labels, sources)`` where *labels* is an ordered list of matched
        topic names and *sources* maps each topic name to its match source.
    """
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
    """Build a compact, numbered context block for LLM summarization prompts.

    Each record is formatted as two lines:

    .. code-block:: text

        [1] PMG22_WAI_q5a | Do you currently have a financial plan?
             survey=PMG22_WAI, wave=2022, topics=Financial Planning (Both), values=1: Yes | 2: No

    The format keeps the context concise while preserving all metadata needed
    for grounded synthesis and citation generation.

    Parameters
    ----------
    records:
        Ordered list of :class:`~backend.models.QuestionRecord` to include.

    Returns
    -------
    str
        Multi-line context block ready to be injected into an LLM prompt.
    """
    lines: list[str] = []
    for idx, record in enumerate(records, start=1):
        lines.append(f"[{idx}] {record.question_id} | {record.question_text}")
        lines.append(
            "     "
            f"survey={record.survey_name}, wave={record.wave_year}, "
            f"topics={record.topic_sources_text}, values={record.value_labels_text}"
        )
    return "\n".join(lines)
