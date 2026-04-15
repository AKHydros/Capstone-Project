"""Tests for multi-turn consistency helpers in ChatbotService.

Covers:
- _query_token_set() produces correct frozensets
- _find_prior_answer() returns prior answer on duplicate query
- _find_prior_answer() returns None for distinct queries
- _find_prior_answer() requires an assistant turn to follow the matched user turn
- Jaccard threshold behaviour (near-duplicate vs distinct)
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.services.chatbot_service import _query_token_set, _QUERY_STOP_WORDS


class QueryTokenSetTests(unittest.TestCase):
    """_query_token_set() tokenisation helper."""

    def test_strips_stop_words(self) -> None:
        """Stop words removed, only content tokens remain."""
        tokens = _query_token_set("what are the options for q5a")
        # "what", "are", "the", "for" are stop words; "options", "q5a" are not
        self.assertIn("options", tokens)
        self.assertIn("q5a", tokens)
        self.assertNotIn("what", tokens)
        self.assertNotIn("the", tokens)

    def test_short_tokens_filtered(self) -> None:
        """Tokens shorter than 3 characters are dropped."""
        tokens = _query_token_set("q5 in wave")
        # "q5" is 2 chars → dropped; "wave" stays
        self.assertNotIn("q5", tokens)
        self.assertIn("wave", tokens)

    def test_empty_query_returns_empty_frozenset(self) -> None:
        """Empty / whitespace query → empty frozenset."""
        self.assertEqual(_query_token_set(""), frozenset())
        self.assertEqual(_query_token_set("   "), frozenset())

    def test_all_stop_words_returns_empty_frozenset(self) -> None:
        """Query consisting only of stop words → empty frozenset."""
        tokens = _query_token_set("what is the")
        self.assertEqual(tokens, frozenset())

    def test_returns_frozenset(self) -> None:
        """Return type is frozenset."""
        result = _query_token_set("PMG22_WAI_q5a financial planning")
        self.assertIsInstance(result, frozenset)


class FindPriorAnswerTests(unittest.TestCase):
    """ChatbotService._find_prior_answer() duplicate-detection logic."""

    def _make_service(self) -> object:
        """Build a minimal ChatbotService with mocked dependencies."""
        from backend.services.chatbot_service import ChatbotService
        retriever = MagicMock()
        retriever.records = []
        llm_client = MagicMock()
        llm_client.is_enabled.return_value = False
        service = object.__new__(ChatbotService)
        service.retriever = retriever
        service.llm_client = llm_client
        service.doc_question_hints = {}
        service.rag_enabled_override = None
        service.rag_confidence_threshold_override = None
        service.rag_score_gap_threshold_override = None
        service.rag_answer_cache_ttl_override = None
        # Minimal post-init setup
        from backend.cache.ttl_cache import TTLCache
        service.answer_cache = TTLCache(ttl_seconds=60, max_size=10)
        service.record_index = {}
        service.rag_enabled = True
        service.rag_confidence_threshold = 0.72
        service.rag_score_gap_threshold = 0.07
        return service

    def test_identical_query_returns_prior_answer(self) -> None:
        """Exact repeat of a prior user turn returns that turn's assistant answer."""
        service = self._make_service()
        context = [
            {"role": "user", "content": "What are the value labels for PMG22_WAI_q5a?"},
            {"role": "assistant", "content": "The value labels are 1=Yes, 2=No."},
        ]
        result = service._find_prior_answer(
            "What are the value labels for PMG22_WAI_q5a?", context
        )
        self.assertEqual(result, "The value labels are 1=Yes, 2=No.")

    def test_near_duplicate_query_returns_prior_answer(self) -> None:
        """High-overlap paraphrase returns prior assistant answer."""
        service = self._make_service()
        context = [
            {"role": "user", "content": "Show me value labels for PMG22_WAI_q5a survey"},
            {"role": "assistant", "content": "PMG22_WAI_q5a has options 1=Yes, 2=No."},
        ]
        # "value labels pmg22_wai_q5a survey" vs "value labels pmg22_wai_q5a wave"
        # Both share "value", "labels", "pmg22_wai_q5a" → Jaccard well above 0.80
        result = service._find_prior_answer(
            "value labels for PMG22_WAI_q5a survey wave", context
        )
        self.assertIsNotNone(result)

    def test_distinct_query_returns_none(self) -> None:
        """Unrelated query returns None — no false duplicate detection."""
        service = self._make_service()
        context = [
            {"role": "user", "content": "What are the value labels for PMG22_WAI_q5a?"},
            {"role": "assistant", "content": "The value labels are 1=Yes, 2=No."},
        ]
        result = service._find_prior_answer(
            "How many respondents participated in the 2022 wave?", context
        )
        self.assertIsNone(result)

    def test_returns_none_when_no_assistant_follows(self) -> None:
        """Matched user turn with no subsequent assistant turn → None."""
        service = self._make_service()
        context = [
            {"role": "user", "content": "What are the value labels for PMG22_WAI_q5a?"},
            # No assistant turn follows
        ]
        result = service._find_prior_answer(
            "What are the value labels for PMG22_WAI_q5a?", context
        )
        self.assertIsNone(result)

    def test_empty_context_returns_none(self) -> None:
        """Empty conversation context → None."""
        service = self._make_service()
        result = service._find_prior_answer("any query here", [])
        self.assertIsNone(result)

    def test_only_stop_words_query_returns_none(self) -> None:
        """Query with no content tokens (all stop words) → None (safe no-op)."""
        service = self._make_service()
        context = [
            {"role": "user", "content": "what is the"},
            {"role": "assistant", "content": "something"},
        ]
        result = service._find_prior_answer("what is the", context)
        self.assertIsNone(result)

    def test_multi_turn_context_returns_first_match_in_order(self) -> None:
        """When multiple matching turns exist, the first match in iteration order is used."""
        service = self._make_service()
        # Both prior user turns share enough tokens with the query to exceed threshold.
        # Turn 0: "value labels PMG22_WAI_q5a survey wave"  → same tokens as query → Jaccard=1.0
        # Turn 2: "value labels PMG22_WAI_q5a survey wave version2" → slightly different
        context = [
            {"role": "user", "content": "value labels PMG22_WAI_q5a survey wave"},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "value labels PMG22_WAI_q5a survey wave version2"},
            {"role": "assistant", "content": "Second answer."},
        ]
        result = service._find_prior_answer(
            "value labels PMG22_WAI_q5a survey wave", context
        )
        # First matching user turn (index 0) has Jaccard=1.0 → returns "First answer."
        self.assertEqual(result, "First answer.")


if __name__ == "__main__":
    unittest.main()
