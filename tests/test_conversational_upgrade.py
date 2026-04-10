from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from backend.config import AppConfig, resolve_api_role
from backend.observability.db import ObservabilityStore
from ui.chat_utils import build_conversation_context, chat_history_markdown, citation_markers, confidence_badge


class ConversationalUtilityTests(unittest.TestCase):
    def test_build_conversation_context_windows_to_last_n_turns(self) -> None:
        """Ensure memory context keeps latest turns and filters unsupported roles."""
        history: list[dict[str, object]] = []
        for idx in range(15):
            history.append({"role": "user", "content": f"question {idx}"})
            history.append({"role": "assistant", "answer": f"answer {idx}"})
        history.append({"role": "system", "content": "ignored"})

        context = build_conversation_context(history, max_turns=20)
        self.assertEqual(len(context), 20)
        self.assertEqual(context[0]["content"], "question 5")
        self.assertEqual(context[-1]["content"], "answer 14")
        self.assertTrue(all(item["role"] in {"user", "assistant"} for item in context))

    def test_markdown_and_citation_helpers(self) -> None:
        """Ensure markdown export includes source rows and marker compacting."""
        history = [
            {"role": "user", "content": "What are q5 options?"},
            {
                "role": "assistant",
                "answer": "Use q5a for the detailed breakdown.",
                "meta": "route=deterministic",
                "sources": [
                    {"marker": "[1]", "label": "PMG20_GAM_q5a | PMG20_GAM 2020", "question_text": "Bank usage"}
                ],
            },
        ]
        rendered = chat_history_markdown(history)
        self.assertIn("Chat Export", rendered)
        self.assertIn("PMG20_GAM_q5a", rendered)
        self.assertEqual(citation_markers(history[1]["sources"]), "[1]")
        self.assertIn("High", confidence_badge(0.81))
        # 0.12 is below the 0.25 floor so the badge now reads "Very low"
        self.assertIn("low", confidence_badge(0.12).lower())


class RoleAndAnalyticsTests(unittest.TestCase):
    def test_role_resolution_from_token_mapping(self) -> None:
        """Verify token-role mapping with shared-token admin fallback."""
        config = AppConfig(
            excel_source_path=Path("data/main.xlsx"),
            index_cache_dir=Path("data/cache"),
            observability_db_path=Path("data/observability.db"),
            logs_dir=Path("data/logs"),
            api_internal_token="internal-admin",
            api_viewer_tokens=("viewer-a",),
            api_analyst_tokens=("analyst-a",),
            api_admin_tokens=("admin-a",),
            api_base_url="",
            governance_policy_version="v1",
            default_router_mode="hybrid",
            conversation_memory_turns=20,
            openai_chat_model="gpt-4.1-mini",
            openai_chat_models=("gpt-4.1-mini",),
            rag_enabled=True,
            rag_confidence_threshold=0.72,
            rag_score_gap_threshold=0.07,
            rag_embed_cache_ttl=1800,
            rag_answer_cache_ttl=600,
            rag_batch_rebuild=True,
        )
        self.assertEqual(resolve_api_role(config, "viewer-a"), "viewer")
        self.assertEqual(resolve_api_role(config, "analyst-a"), "analyst")
        self.assertEqual(resolve_api_role(config, "admin-a"), "admin")
        self.assertEqual(resolve_api_role(config, "internal-admin"), "admin")
        self.assertIsNone(resolve_api_role(config, "unknown"))

    def test_unanswered_analytics_aggregates_reason_counts(self) -> None:
        """Ensure unanswered analytics reports counts and repeated patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ObservabilityStore(Path(temp_dir) / "obs.db")
            store.record_unanswered(
                trace_id="t1",
                session_id="s1",
                user_role="analyst",
                reason="no_cards",
                query_text="Question with no match",
            )
            store.record_unanswered(
                trace_id="t2",
                session_id="s1",
                user_role="analyst",
                reason="clarifier_only",
                query_text="Need q5a or q5b?",
            )
            store.record_unanswered(
                trace_id="t3",
                session_id="s2",
                user_role="viewer",
                reason="no_cards",
                query_text="Question with no match",
            )
            payload = store.unanswered_analytics(limit=10)
            self.assertEqual(payload["counts"]["no_cards"], 2)
            self.assertEqual(payload["counts"]["clarifier_only"], 1)
            self.assertGreaterEqual(len(payload["recent"]), 3)
            self.assertTrue(any(item["count"] >= 2 for item in payload["top_patterns"]))


if __name__ == "__main__":
    unittest.main()
