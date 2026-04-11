from __future__ import annotations

from dataclasses import dataclass
import re

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
        """Builds chunks + lexical + semantic retrievers from records."""
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
        """Runs weighted lexical+semantic ranking, filtering, and returns diagnostics.

        When ``survey_name`` is provided, record indices are pre-filtered before
        the chunk-scoring loop so chunks belonging to other surveys are skipped
        entirely.  This reduces the effective loop size from ``len(chunks)`` to
        ``len(survey_chunks)`` — a significant win when most queries are scoped
        to a single survey.

        Diagnostics (``top_score``, ``score_gap``) are computed from pre-filter
        scores, then ``apply_filters`` enforces additional wave/topic constraints.
        """
        top_k = top_k or RETRIEVAL_RULES.top_k

        # Pre-compute valid record indices for survey filter — O(n) once, saves
        # O(n) filter work inside the chunk loop.
        if survey_name:
            survey_upper = survey_name.upper()
            valid_record_indices: frozenset[int] | None = frozenset(
                idx for idx, r in enumerate(self.records)
                if r.survey_name.upper() == survey_upper
            )
        else:
            valid_record_indices = None

        lexical_scores = self.lexical.score(query)
        exact_match_boosts = self._exact_match_boosts(query)
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
            # Skip chunks outside the active survey filter — avoids scoring
            # irrelevant records when survey_name is set.
            if valid_record_indices is not None and chunk.record_index not in valid_record_indices:
                continue
            if idx >= len(lexical_scores) or idx >= len(semantic_scores):
                continue
            score = (
                lexical_weight * lexical_scores[idx]
                + semantic_weight * semantic_scores[idx]
            )
            score += exact_match_boosts.get(chunk.record_index, 0.0)
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
        """Return ranked :class:`~backend.models.QuestionRecord` objects for *query*.

        Convenience wrapper around :meth:`search_with_details` that discards
        diagnostics and returns only the ordered record list.

        Parameters
        ----------
        query:
            User query string.
        survey_name:
            Optional survey-name filter applied after scoring.
        wave_year:
            Optional wave/year filter applied after scoring.
        topic_label:
            Optional topic-label filter applied after scoring.
        topic_source_type:
            Optional topic-source-type filter applied after scoring.
        top_k:
            Maximum number of records to return.  Defaults to
            ``RETRIEVAL_RULES.top_k``.

        Returns
        -------
        list[QuestionRecord]
            Ordered from highest to lowest relevance score.
        """
        result = self.search_with_details(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
            top_k=top_k,
        )
        return [item.record for item in result.scored_results]

    def _exact_match_boosts(self, query: str) -> dict[int, float]:
        """Computes deterministic score boosts for exact variable/question references in the query.

        Returns an empty dict immediately when the query contains no survey token
        (e.g. ``PMG18_ROB``) and no question reference (e.g. ``q5``), avoiding
        an O(n) record scan that would contribute nothing.
        """
        lowered = query.lower()

        # Fast pre-check: skip full record scan when query has no IDs to match.
        # Three patterns cover all reference forms:
        #   full variable ID  — PMG20_GAM_q12a  (\b only at token start; _q suffix present)
        #   standalone survey — PMG20_GAM        (\b required on both ends)
        #   question ref      — q5 / question 5  (\b required before q)
        has_full_id = bool(re.search(r"\b[a-z]{3}\d{2}_[a-z]{3}_q{1,2}\d+", lowered))
        has_survey_token = bool(re.search(r"\b[a-z]{3}\d{2}_[a-z]{3}\b", lowered))
        has_question_token = bool(re.search(r"\bq(?:uestion)?\s*\d+", lowered))
        if not has_full_id and not has_survey_token and not has_question_token:
            return {}

        boosts: dict[int, float] = {}

        full_id_matches = set(
            re.findall(r"\b[a-z]{3}\d{2}_[a-z]{3}_q{1,2}\d+(?:\.\d+)?[a-z]?\b", lowered)
        )

        survey_tokens = {
            f"{prefix}{year}_{suffix}"
            for prefix, year, suffix in re.findall(r"\b([a-z]{3})[\s_-]?(\d{2})[\s_-]?([a-z]{3})\b", lowered)
        }
        question_tokens = {
            token
            for token in re.findall(r"\bq(?:uestion)?\s*(\d+(?:\.\d+)?[a-z]?)\b", lowered)
        }

        for idx, record in enumerate(self.records):
            record_id = record.question_id.lower()
            if record_id in full_id_matches:
                boosts[idx] = max(boosts.get(idx, 0.0), 0.45)
                continue

            if not survey_tokens or not question_tokens:
                continue
            if record.survey_name.lower() not in survey_tokens:
                continue

            component_match = re.search(r"_q{1,2}(\d+(?:\.\d+)?[a-z]?)\b", record_id)
            if not component_match:
                continue
            if component_match.group(1) in question_tokens:
                boosts[idx] = max(boosts.get(idx, 0.0), 0.35)

        return boosts
