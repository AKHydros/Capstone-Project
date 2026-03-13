from __future__ import annotations

from dataclasses import dataclass

from ..business_rules import RETRIEVAL_RULES, apply_filters
from ..models import QuestionRecord
from .lexical import LexicalRetriever
from .semantic import SemanticRetriever


@dataclass
class HybridRetriever:
    records: list[QuestionRecord]
    lexical: LexicalRetriever
    semantic: SemanticRetriever

    @classmethod
    def build(cls, records: list[QuestionRecord]) -> "HybridRetriever":
        return cls(
            records=records,
            lexical=LexicalRetriever.build(records),
            semantic=SemanticRetriever.build(records),
        )

    def search(
        self,
        query: str,
        survey_name: str | None = None,
        wave_year: str | None = None,
        topic_label: str | None = None,
        topic_source_type: str | None = None,
        top_k: int | None = None,
    ) -> list[QuestionRecord]:
        top_k = top_k or RETRIEVAL_RULES.top_k
        lexical_scores = self.lexical.score(query)
        semantic_scores = self.semantic.score(query)

        scored: list[tuple[float, QuestionRecord]] = []
        for idx, record in enumerate(self.records):
            if idx >= len(lexical_scores) or idx >= len(semantic_scores):
                continue
            score = (
                RETRIEVAL_RULES.lexical_weight * lexical_scores[idx]
                + RETRIEVAL_RULES.semantic_weight * semantic_scores[idx]
            )
            if score >= RETRIEVAL_RULES.min_score_threshold:
                scored.append((score, record))

        filtered_records = apply_filters(
            (r for _, r in scored),
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        score_map = {r.question_id: s for s, r in scored}
        ranked = sorted(filtered_records, key=lambda r: score_map.get(r.question_id, 0.0), reverse=True)
        return ranked[:top_k]
