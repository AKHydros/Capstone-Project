from __future__ import annotations

import unittest

from backend.models import QuestionRecord
from backend.retrieval.hybrid import HybridSearchResult, RetrievalDiagnostics, ScoredRecord
from backend.services.chatbot_service import ChatbotService


class _DummyLlmClient:
    enabled = False

    def summarize(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("LLM summarize should not be called in these tests")


class _FakeRetriever:
    def __init__(self, records: list[QuestionRecord]) -> None:
        self.records = records
        self.search_with_details_calls = 0
        self.search_calls = 0

    def search_with_details(
        self,
        query: str,
        survey_name: str | None = None,
        wave_year: str | None = None,
        topic_label: str | None = None,
        topic_source_type: str | None = None,
        top_k: int | None = None,
    ) -> HybridSearchResult:
        del query, survey_name, wave_year, topic_label, topic_source_type, top_k
        self.search_with_details_calls += 1
        scored = [ScoredRecord(record=record, score=max(0.1, 1.0 - (idx * 0.05))) for idx, record in enumerate(self.records)]
        diagnostics = RetrievalDiagnostics(
            top_score=0.92,
            second_score=0.64,
            score_gap=0.28,
            embedding_cache_hit=False,
            candidate_count=len(scored),
            returned_count=len(scored),
        )
        return HybridSearchResult(scored_results=scored, diagnostics=diagnostics)

    def search(
        self,
        query: str,
        survey_name: str | None = None,
        wave_year: str | None = None,
        topic_label: str | None = None,
        topic_source_type: str | None = None,
        top_k: int | None = None,
    ) -> list[QuestionRecord]:
        del query, survey_name, wave_year, topic_label, topic_source_type
        self.search_calls += 1
        return self.records[: top_k or len(self.records)]


def _record(question_id: str, text: str, values: list[str]) -> QuestionRecord:
    return QuestionRecord(
        question_id=question_id,
        question_text=text,
        measurement_level="Ordinal",
        role="Input",
        source_file="data/fixture.xlsx",
        survey_name="PMG20_GAM",
        wave_year="2020",
        value_labels=values,
        topic_labels=["General"],
        topic_label_sources={"General": "Fallback"},
    )


class ChatbotServiceDeterministicLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _record("PMG20_GAM_q11a", "Banks", []),
            _record("PMG20_GAM_qq11a", "Banks", ["1: Yes", "2: No", "98: Not sure"]),
            _record("PMG20_GAM_q11b", "Online banks", []),
            _record("PMG20_GAM_qq11b", "Online banks", ["1: Yes", "2: No"]),
            _record("PMG20_GAM_q12a", "Consolidated number of financial companies you use", ["1: Yes", "2: No"]),
            _record("PMG20_GAM_q12b", "Switched financial companies", ["1: Yes", "2: No"]),
            _record("PMG20_GAM_q16", "Likelihood to continue using primary institution", []),
        ]
        self.retriever = _FakeRetriever(self.records)
        self.service = ChatbotService(
            retriever=self.retriever,
            llm_client=_DummyLlmClient(),
        )

    def test_question_11_options_returns_variant_clarification(self) -> None:
        response = self.service.chat("What are the dropdown options for PMG20_GAM question 11?")
        self.assertIn("multiple variants", response.answer.lower())
        self.assertIn("`11a`", response.answer)
        self.assertIn("`11b`", response.answer)
        self.assertEqual(response.lookup_mode, "exact_id")
        self.assertEqual(response.variant_count, 2)
        self.assertEqual(self.retriever.search_with_details_calls, 0)

    def test_question_11a_options_uses_alias_backed_values(self) -> None:
        response = self.service.chat("What are the dropdown options for PMG20_GAM question 11a?")
        self.assertIn("allowable response options", response.answer.lower())
        self.assertIn("- 1: Yes", response.answer)
        self.assertIn("- 2: No", response.answer)
        self.assertIn("Matched variable IDs", response.answer)
        self.assertIn("`PMG20_GAM_q11a`", response.answer)
        self.assertIn("`PMG20_GAM_qq11a`", response.answer)
        self.assertEqual(response.lookup_mode, "exact_id")
        self.assertEqual(response.variant_count, 1)

    def test_question_12_options_returns_variant_clarification(self) -> None:
        response = self.service.chat("What are the dropdown options for PMG20_GAM question 12?")
        self.assertIn("multiple variants", response.answer.lower())
        self.assertIn("`12a`", response.answer)
        self.assertIn("`12b`", response.answer)
        self.assertEqual(response.lookup_mode, "exact_id")
        self.assertEqual(response.variant_count, 2)

    def test_question_12a_options_returns_values(self) -> None:
        response = self.service.chat("What are the dropdown options for PMG20_GAM question 12a?")
        self.assertIn("allowable response options", response.answer.lower())
        self.assertIn("- 1: Yes", response.answer)
        self.assertIn("- 2: No", response.answer)
        self.assertEqual(response.lookup_mode, "exact_id")
        self.assertEqual(response.variant_count, 1)

    def test_non_id_query_uses_hybrid_fallback(self) -> None:
        response = self.service.chat("How likely are respondents to continue with their primary institution?")
        self.assertEqual(response.lookup_mode, "hybrid_fallback")
        self.assertEqual(response.variant_count, 0)
        self.assertGreater(self.retriever.search_with_details_calls, 0)
        self.assertTrue(response.ranked_results)


if __name__ == "__main__":
    unittest.main()
