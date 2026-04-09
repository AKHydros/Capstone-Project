from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.models import QuestionRecord
from backend.retrieval.hybrid import HybridRetriever
from backend.services.chatbot_service import ChatbotService


class StubLlmClient:
    def __init__(self) -> None:
        """Enable LLM mode in tests while counting summarize calls for assertions."""
        self.enabled = True
        self.calls = 0

    def summarize(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        """Return a fixed summary string and increment the stub call counter."""
        self.calls += 1
        return "LLM summary"


def make_records() -> list[QuestionRecord]:
    """Build two survey records used across hybrid and RAG behavior tests."""
    return [
        QuestionRecord(
            question_id="PMG22_WAI_q1",
            question_text="What is your annual household income?",
            measurement_level="Ordinal",
            role="Respondent",
            source_file="test.xlsx",
            survey_name="PMG22_WAI",
            wave_year="2022",
            value_labels=["1: <50k", "2: 50k-100k", "3: 100k+"],
            topic_labels=["Demographics"],
            topic_label_sources={"Demographics": "Question Text"},
        ),
        QuestionRecord(
            question_id="PMG22_WAI_q2",
            question_text="How often do you use mobile banking?",
            measurement_level="Ordinal",
            role="Respondent",
            source_file="test.xlsx",
            survey_name="PMG22_WAI",
            wave_year="2022",
            value_labels=["1: Never", "2: Sometimes", "3: Often"],
            topic_labels=["Digital Behavior"],
            topic_label_sources={"Digital Behavior": "Question Text"},
        ),
    ]


class RagUpgradeTests(unittest.TestCase):
    def test_hybrid_search_returns_stable_top_result(self) -> None:
        """Assert repeated hybrid queries keep the same top-ranked record."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            retriever = HybridRetriever.build(make_records())

        first = retriever.search_with_details("annual household income", survey_name="PMG22_WAI")
        second = retriever.search_with_details("annual household income", survey_name="PMG22_WAI")

        self.assertGreaterEqual(len(first.scored_results), 1)
        self.assertEqual(first.scored_results[0].record.question_id, "PMG22_WAI_q1")
        self.assertEqual(second.scored_results[0].record.question_id, "PMG22_WAI_q1")
        self.assertGreaterEqual(first.diagnostics.top_score, first.diagnostics.second_score)

    def test_confidence_gate_skips_llm_for_high_confidence_queries(self) -> None:
        """Assert confidence gates keep response deterministic and skip LLM calls."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            retriever = HybridRetriever.build(make_records())
        llm = StubLlmClient()

        with patch.dict(
            os.environ,
            {
                "RAG_ENABLED": "true",
                "RAG_CONFIDENCE_THRESHOLD": "0.05",
                "RAG_SCORE_GAP_THRESHOLD": "0.01",
                "RAG_ANSWER_CACHE_TTL": "600",
            },
            clear=False,
        ):
            service = ChatbotService(retriever=retriever, llm_client=llm)
            response = service.chat(
                "annual household income",
                survey_name="PMG22_WAI",
                use_llm=True,
                llm_provider="chatgpt",
            )

        self.assertEqual(response.retrieval_mode, "deterministic")
        self.assertEqual(llm.calls, 0)

    def test_llm_synthesis_uses_answer_cache_on_repeated_queries(self) -> None:
        """Assert repeated synthesis queries hit answer cache after first LLM call."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            retriever = HybridRetriever.build(make_records())
        llm = StubLlmClient()

        with patch.dict(
            os.environ,
            {
                "RAG_ENABLED": "true",
                "RAG_CONFIDENCE_THRESHOLD": "0.95",
                "RAG_SCORE_GAP_THRESHOLD": "0.5",
                "RAG_ANSWER_CACHE_TTL": "600",
            },
            clear=False,
        ):
            service = ChatbotService(retriever=retriever, llm_client=llm)
            first = service.chat(
                "summarize annual household income",
                survey_name="PMG22_WAI",
                use_llm=True,
                llm_provider="chatgpt",
            )
            second = service.chat(
                "summarize annual household income",
                survey_name="PMG22_WAI",
                use_llm=True,
                llm_provider="chatgpt",
            )

        self.assertEqual(first.retrieval_mode, "llm_synthesized")
        self.assertFalse(first.answer_cache_hit)
        self.assertTrue(second.answer_cache_hit)
        self.assertEqual(llm.calls, 1)


if __name__ == "__main__":
    unittest.main()
