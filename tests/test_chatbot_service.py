from __future__ import annotations

import unittest

from backend.models import QuestionRecord
from backend.retrieval.hybrid import HybridSearchResult, RetrievalDiagnostics, ScoredRecord
from backend.services.chatbot_service import ChatbotService


class _DummyLlmClient:
    enabled = False

    def summarize(self, *args, **kwargs):  # pragma: no cover
        """Fail fast if a test unexpectedly tries to call LLM summarization."""
        raise AssertionError("LLM summarize should not be called in these tests")


class _FakeRetriever:
    def __init__(self, records: list[QuestionRecord]) -> None:
        """Store fixture records and initialize call counters for assertions."""
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
        """Return deterministic scored results and diagnostics for service-level tests."""
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
        """Return fixture records while tracking fallback lexical search invocations."""
        del query, survey_name, wave_year, topic_label, topic_source_type
        self.search_calls += 1
        return self.records[: top_k or len(self.records)]


def _record(
    question_id: str,
    text: str,
    values: list[str],
    *,
    survey_name: str = "PMG20_GAM",
    wave_year: str = "2020",
    context_fields: dict[str, str] | None = None,
) -> QuestionRecord:
    """Build a standardized `QuestionRecord` fixture for PMG20_GAM scenarios."""
    return QuestionRecord(
        question_id=question_id,
        question_text=text,
        measurement_level="Ordinal",
        role="Input",
        source_file="data/fixture.xlsx",
        survey_name=survey_name,
        wave_year=wave_year,
        value_labels=values,
        topic_labels=["General"],
        topic_label_sources={"General": "Fallback"},
        context_fields=context_fields or {},
    )


class ChatbotServiceDeterministicLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create shared fixtures covering aliased and multi-variant question IDs."""
        self.records = [
            _record("PMG20_GAM_q11a", "Banks", []),
            _record("PMG20_GAM_qq11a", "Banks", ["1: Yes", "2: No", "98: Not sure"]),
            _record("PMG20_GAM_q11b", "Online banks", []),
            _record("PMG20_GAM_qq11b", "Online banks", ["1: Yes", "2: No"]),
            _record("PMG20_GAM_q12a", "Consolidated number of financial companies you use", ["1: Yes", "2: No"]),
            _record("PMG20_GAM_q12b", "Switched financial companies", ["1: Yes", "2: No"]),
            _record("PMG20_GAM_q16", "Likelihood to continue using primary institution", []),
            _record(
                "PMG19_GAM_q1.1",
                "Market research",
                [],
                survey_name="PMG19_GAM",
                wave_year="2019",
                context_fields={
                    "position": "20",
                    "label": "Market research",
                    "measurement_level": "Nominal",
                    "role": "Input",
                    "column_width": "5",
                    "alignment": "Right",
                    "print_format": "F12",
                    "write_format": "F12",
                },
            ),
        ]
        self.retriever = _FakeRetriever(self.records)
        self.service = ChatbotService(
            retriever=self.retriever,
            llm_client=_DummyLlmClient(),
        )

    def test_question_11_options_returns_variant_clarification(self) -> None:
        """Assert broad `question 11` lookup prompts user to pick a specific variant."""
        response = self.service.chat("What are the dropdown options for PMG20_GAM question 11?")
        self.assertIn("multiple variants", response.answer.lower())
        self.assertIn("`11a`", response.answer)
        self.assertIn("`11b`", response.answer)
        self.assertEqual(response.lookup_mode, "exact_id")
        self.assertEqual(response.variant_count, 2)
        self.assertEqual(response.answer_mode, "clarifier")
        self.assertTrue(response.needs_clarification)
        self.assertEqual(self.retriever.search_with_details_calls, 0)

    def test_question_11a_options_uses_alias_backed_values(self) -> None:
        """Assert specific `question 11a` lookup merges alias-backed dropdown values."""
        response = self.service.chat("What are the dropdown options for PMG20_GAM question 11a?")
        self.assertIn("allowable response options", response.answer.lower())
        self.assertIn("- 1: Yes", response.answer)
        self.assertIn("- 2: No", response.answer)
        self.assertIn("Matched variable IDs", response.answer)
        self.assertIn("`PMG20_GAM_q11a`", response.answer)
        self.assertIn("`PMG20_GAM_qq11a`", response.answer)
        self.assertEqual(response.lookup_mode, "exact_id")
        self.assertEqual(response.variant_count, 1)
        self.assertEqual(response.answer_mode, "allowed_values")
        self.assertFalse(response.needs_clarification)

    def test_question_12_options_returns_variant_clarification(self) -> None:
        """Assert broad `question 12` lookup returns a variant disambiguation prompt."""
        response = self.service.chat("What are the dropdown options for PMG20_GAM question 12?")
        self.assertIn("multiple variants", response.answer.lower())
        self.assertIn("`12a`", response.answer)
        self.assertIn("`12b`", response.answer)
        self.assertEqual(response.lookup_mode, "exact_id")
        self.assertEqual(response.variant_count, 2)
        self.assertEqual(response.answer_mode, "clarifier")
        self.assertTrue(response.needs_clarification)

    def test_question_12a_options_returns_values(self) -> None:
        """Assert specific `question 12a` lookup returns coded allowable values."""
        response = self.service.chat("What are the dropdown options for PMG20_GAM question 12a?")
        self.assertIn("allowable response options", response.answer.lower())
        self.assertIn("- 1: Yes", response.answer)
        self.assertIn("- 2: No", response.answer)
        self.assertEqual(response.lookup_mode, "exact_id")
        self.assertEqual(response.variant_count, 1)
        self.assertEqual(response.answer_mode, "allowed_values")
        self.assertFalse(response.needs_clarification)

    def test_non_id_query_uses_hybrid_fallback(self) -> None:
        """Assert non-ID queries bypass exact lookup and use hybrid retrieval fallback."""
        response = self.service.chat("How likely are respondents to continue with their primary institution?")
        self.assertEqual(response.lookup_mode, "hybrid_fallback")
        self.assertEqual(response.variant_count, 0)
        self.assertEqual(response.answer_mode, "direct_answer")
        self.assertFalse(response.needs_clarification)
        self.assertGreater(self.retriever.search_with_details_calls, 0)
        self.assertTrue(response.ranked_results)

    def test_decimal_question_context_returns_exact_values(self) -> None:
        """Assert decimal question refs return exact context fields from dictionary metadata."""
        response = self.service.chat("For PMG19_GAM_q1.1 what is the position and label of the data?")
        self.assertIn("exact dictionary context", response.answer.lower())
        self.assertIn("- Position: 20", response.answer)
        self.assertIn("- Label: Market research", response.answer)
        self.assertEqual(response.lookup_mode, "exact_id")
        self.assertEqual(response.variant_count, 1)
        self.assertEqual(self.retriever.search_with_details_calls, 0)

    def test_exact_context_lookup_uses_cache_on_repeat(self) -> None:
        """Assert repeated exact context lookups are served from cache."""
        first = self.service.chat("For PMG19_GAM_q1.1 what is the position and label of the data?")
        second = self.service.chat("For PMG19_GAM_q1.1 what is the position and label of the data?")
        self.assertFalse(first.answer_cache_hit)
        self.assertTrue(second.answer_cache_hit)
        self.assertEqual(first.answer, second.answer)
        self.assertEqual(self.retriever.search_with_details_calls, 0)


if __name__ == "__main__":
    unittest.main()
