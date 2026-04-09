from __future__ import annotations

import unittest

from backend.models import QuestionRecord
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.pipeline import TextChunk


class _FailingSemantic:
    def score_with_meta(self, query: str) -> tuple[list[float], bool]:
        """Simulate a semantic backend outage by raising at scoring time."""
        del query
        raise RuntimeError("semantic backend unavailable")


class _StaticLexical:
    def __init__(self, scores: list[float]) -> None:
        """Store fixed lexical scores so ranking behavior is deterministic."""
        self._scores = scores

    def score(self, query: str) -> list[float]:
        """Return preconfigured lexical scores regardless of query content."""
        del query
        return self._scores


def _record(question_id: str, text: str) -> QuestionRecord:
    """Build a minimal `QuestionRecord` fixture for retrieval fallback testing."""
    return QuestionRecord(
        question_id=question_id,
        question_text=text,
        measurement_level="Ordinal",
        role="Input",
        source_file="data/fixture.xlsx",
        survey_name="PMG20_GAM",
        wave_year="2020",
        value_labels=[],
        topic_labels=["General"],
        topic_label_sources={"General": "Fallback"},
    )


class HybridRetrieverFailureFallbackTests(unittest.TestCase):
    def test_semantic_failure_falls_back_to_lexical_scores(self) -> None:
        """Assert lexical scores still rank results when semantic scoring fails."""
        records = [
            _record("PMG20_GAM_q12a", "Consolidated number of financial companies you use"),
            _record("PMG20_GAM_q12b", "Switched financial companies"),
        ]
        chunks = [
            TextChunk(chunk_id="PMG20_GAM_q12a:0", record_index=0, question_id=records[0].question_id, text="companies use"),
            TextChunk(chunk_id="PMG20_GAM_q12b:0", record_index=1, question_id=records[1].question_id, text="switched companies"),
        ]

        retriever = HybridRetriever(
            records=records,
            chunks=chunks,
            lexical=_StaticLexical([0.90, 0.40]),
            semantic=_FailingSemantic(),
        )

        detailed = retriever.search_with_details("companies")
        self.assertEqual(len(detailed.scored_results), 2)
        self.assertEqual(detailed.scored_results[0].record.question_id, "PMG20_GAM_q12a")
        self.assertFalse(detailed.diagnostics.embedding_cache_hit)

        ranked = retriever.search("companies")
        self.assertEqual([item.question_id for item in ranked], ["PMG20_GAM_q12a", "PMG20_GAM_q12b"])


if __name__ == "__main__":
    unittest.main()
