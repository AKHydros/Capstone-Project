from __future__ import annotations

from dataclasses import dataclass

from ..business_rules import RETRIEVAL_RULES, apply_filters
from ..models import QuestionRecord
from .lexical import LexicalRetriever
from .pipeline import TextChunk, ingest_clean_chunk
from .semantic import SemanticRetriever


@dataclass(frozen=True)
class ScoredRecord:
    record: QuestionRecord
    score: float


@dataclass(frozen=True)
class RetrievalDiagnostics:
    top_score: float
    second_score: float
    score_gap: float
    embedding_cache_hit: bool
    candidate_count: int
    returned_count: int


@dataclass(frozen=True)
class HybridSearchResult:
    scored_results: list[ScoredRecord]
    diagnostics: RetrievalDiagnostics


@dataclass
class HybridRetriever:
    records: list[QuestionRecord]
    chunks: list[TextChunk]
    lexical: LexicalRetriever
    semantic: SemanticRetriever

    @classmethod
    def build(cls, records: list[QuestionRecord]) -> "HybridRetriever":
        chunks = ingest_clean_chunk(records)
        return cls(
            records=records,
            chunks=chunks,
            lexical=LexicalRetriever.build(chunks),
            semantic=SemanticRetriever.build(records, chunks),
        )

    def search_with_details(
        self,
        query: str,
        survey_name: str | None = None,
        wave_year: str | None = None,
        topic_label: str | None = None,
        topic_source_type: str | None = None,
        top_k: int | None = None,
    ) -> HybridSearchResult:
        top_k = top_k or RETRIEVAL_RULES.top_k
        lexical_scores = self.lexical.score(query)
        try:
            semantic_scores, embedding_cache_hit = self.semantic.score_with_meta(query)
            lexical_weight = RETRIEVAL_RULES.lexical_weight
            semantic_weight = RETRIEVAL_RULES.semantic_weight
        except Exception:  # noqa: BLE001
            semantic_scores = [0.0 for _ in lexical_scores]
            embedding_cache_hit = False
            lexical_weight = 1.0
            semantic_weight = 0.0

        chunk_scores: list[tuple[float, TextChunk]] = []
        for idx, chunk in enumerate(self.chunks):
            if idx >= len(lexical_scores) or idx >= len(semantic_scores):
                continue
            score = (
                lexical_weight * lexical_scores[idx]
                + semantic_weight * semantic_scores[idx]
            )
            if score >= RETRIEVAL_RULES.min_score_threshold:
                chunk_scores.append((score, chunk))

        if not chunk_scores:
            diagnostics = RetrievalDiagnostics(
                top_score=0.0,
                second_score=0.0,
                score_gap=0.0,
                embedding_cache_hit=embedding_cache_hit,
                candidate_count=0,
                returned_count=0,
            )
            return HybridSearchResult(scored_results=[], diagnostics=diagnostics)

        aggregate_scores: dict[int, float] = {}
        for score, chunk in chunk_scores:
            current = aggregate_scores.get(chunk.record_index, float("-inf"))
            if score > current:
                aggregate_scores[chunk.record_index] = score

        scored_records: list[tuple[float, QuestionRecord]] = []
        for record_index, score in aggregate_scores.items():
            if record_index < 0 or record_index >= len(self.records):
                continue
            scored_records.append((score, self.records[record_index]))

        ranked_scores = sorted((score for score, _ in scored_records), reverse=True)
        top_score = ranked_scores[0] if ranked_scores else 0.0
        second_score = ranked_scores[1] if len(ranked_scores) > 1 else 0.0
        score_gap = max(0.0, top_score - second_score)

        filtered_records = apply_filters(
            (r for _, r in scored_records),
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        score_map = {id(record): score for score, record in scored_records}
        ranked = sorted(filtered_records, key=lambda r: score_map.get(id(r), 0.0), reverse=True)
        top_ranked = ranked[:top_k]
        result = [ScoredRecord(record=record, score=score_map.get(id(record), 0.0)) for record in top_ranked]

        diagnostics = RetrievalDiagnostics(
            top_score=top_score,
            second_score=second_score,
            score_gap=score_gap,
            embedding_cache_hit=embedding_cache_hit,
            candidate_count=len(filtered_records),
            returned_count=len(result),
        )
        return HybridSearchResult(scored_results=result, diagnostics=diagnostics)

    def search(
        self,
        query: str,
        survey_name: str | None = None,
        wave_year: str | None = None,
        topic_label: str | None = None,
        topic_source_type: str | None = None,
        top_k: int | None = None,
    ) -> list[QuestionRecord]:
        result = self.search_with_details(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
            top_k=top_k,
        )
        return [item.record for item in result.scored_results]
