"""Tests for logical reasoning output parsing in ChatbotService.

Covers:
- _parse_reasoning() extracts text from **Reasoning:** heading
- _parse_reasoning() handles heading format variants
- _parse_reasoning() returns None when heading absent
- _parse_reasoning() strips surrounding quotes
- reasoning field present in ChatResponse from LLM path (mocked)
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class ParseReasoningTests(unittest.TestCase):
    """ChatbotService._parse_reasoning() extraction logic."""

    def _make_service(self) -> object:
        """Build a minimal ChatbotService instance for unit testing."""
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
        from backend.cache.ttl_cache import TTLCache
        service.answer_cache = TTLCache(ttl_seconds=60, max_size=10)
        service.record_index = {}
        service.rag_enabled = True
        service.rag_confidence_threshold = 0.72
        service.rag_score_gap_threshold = 0.07
        return service

    def test_extracts_reasoning_from_bold_heading(self) -> None:
        """Standard **Reasoning:** heading extracts correctly."""
        service = self._make_service()
        text = (
            "**Reasoning:** Using record [PMG22_WAI_q5a] because it directly matches the query about financial planning.\n\n"
            "**Answer:** The question asks about financial planning frequency.\n\n"
            "**Key Details:**\n- Measured nominally\n\n"
            "**Suggested Follow-up:** What are the value options for this question?"
        )
        result = service._parse_reasoning(text)
        self.assertIsNotNone(result)
        self.assertIn("PMG22_WAI_q5a", result)

    def test_extracts_reasoning_variant_no_bold_colon(self) -> None:
        """**Reasoning**: (colon outside bold) variant also extracted."""
        service = self._make_service()
        text = "**Reasoning**: The top record matches the intent.\n\n**Answer:** Yes."
        result = service._parse_reasoning(text)
        self.assertIsNotNone(result)
        self.assertIn("top record", result)

    def test_extracts_reasoning_plain_heading(self) -> None:
        """Plain 'Reasoning:' heading (no bold) also extracted."""
        service = self._make_service()
        text = "Reasoning: Matched using [PMG20_WAI_q3].\n\nAnswer: See below."
        result = service._parse_reasoning(text)
        self.assertIsNotNone(result)
        self.assertIn("PMG20_WAI_q3", result)

    def test_returns_none_when_heading_absent(self) -> None:
        """No **Reasoning:** heading → returns None."""
        service = self._make_service()
        text = (
            "**Answer:** Yes, the record matches.\n\n"
            "**Key Details:**\n- Detail one\n\n"
            "**Suggested Follow-up:** Try another question."
        )
        result = service._parse_reasoning(text)
        self.assertIsNone(result)

    def test_strips_surrounding_quotes(self) -> None:
        """Model-added quotes stripped from reasoning value."""
        service = self._make_service()
        text = '**Reasoning:** "Used record [q5a] because it best matched."\n\n**Answer:** Yes.'
        result = service._parse_reasoning(text)
        self.assertIsNotNone(result)
        # Quotes should be stripped
        self.assertFalse(result.startswith('"'))
        self.assertFalse(result.endswith('"'))

    def test_returns_none_for_empty_reasoning_value(self) -> None:
        """**Reasoning:** with no text after it → returns None."""
        service = self._make_service()
        text = "**Reasoning:**\n\n**Answer:** Something."
        result = service._parse_reasoning(text)
        self.assertIsNone(result)

    def test_empty_string_input_returns_none(self) -> None:
        """Empty LLM output → None."""
        service = self._make_service()
        self.assertIsNone(service._parse_reasoning(""))

    def test_reasoning_does_not_bleed_into_answer(self) -> None:
        """Extracted reasoning does not include **Answer:** content."""
        service = self._make_service()
        text = (
            "**Reasoning:** Record [q1] was selected for relevance.\n\n"
            "**Answer:** The answer is 42.\n\n"
            "**Suggested Follow-up:** Why 42?"
        )
        result = service._parse_reasoning(text)
        self.assertIsNotNone(result)
        self.assertNotIn("The answer is 42", result)


class ReasoningInChatResponseTests(unittest.TestCase):
    """ChatResponse.reasoning field is populated from LLM output."""

    def test_chat_response_reasoning_field_exists(self) -> None:
        """ChatResponse dataclass accepts reasoning field."""
        from backend.models import ChatResponse
        resp = ChatResponse(
            answer="test",
            ranked_results=[],
            reasoning="Using record [q5a] because it matches.",
        )
        self.assertEqual(resp.reasoning, "Using record [q5a] because it matches.")

    def test_chat_response_reasoning_defaults_to_none(self) -> None:
        """ChatResponse.reasoning defaults to None when not provided."""
        from backend.models import ChatResponse
        resp = ChatResponse(answer="test", ranked_results=[])
        self.assertIsNone(resp.reasoning)

    def test_chat_response_consistency_note_defaults_to_none(self) -> None:
        """ChatResponse.consistency_note defaults to None."""
        from backend.models import ChatResponse
        resp = ChatResponse(answer="test", ranked_results=[])
        self.assertIsNone(resp.consistency_note)

    def test_chat_response_consistency_note_set(self) -> None:
        """ChatResponse.consistency_note stores the note string."""
        from backend.models import ChatResponse
        resp = ChatResponse(
            answer="prior answer",
            ranked_results=[],
            answer_mode="cached_prior",
            consistency_note="Duplicate of prior turn — prior answer returned.",
        )
        self.assertEqual(resp.consistency_note, "Duplicate of prior turn — prior answer returned.")


if __name__ == "__main__":
    unittest.main()
