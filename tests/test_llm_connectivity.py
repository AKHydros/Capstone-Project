from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.llm.key_utils import openai_key_source, resolve_openai_api_key
from backend.llm.openai_client import OpenAIChatClient
from backend.services.llm_health_service import LlmHealthService


class LlmConnectivityTests(unittest.TestCase):
    def test_resolve_openai_key_supports_alias(self) -> None:
        """Accept OPEN_API_KEY as a compatibility alias when OPENAI_API_KEY is empty."""
        with patch("backend.llm.key_utils._dotenv_key_values", return_value={}):
            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "", "OPEN_API_KEY": " sk-test-compat-key "},
                clear=False,
            ):
                self.assertEqual(resolve_openai_api_key(), "sk-test-compat-key")
                self.assertEqual(openai_key_source(), "OPEN_API_KEY")

    def test_openai_key_strips_optional_quotes(self) -> None:
        """Trim accidental quotes around key values from .env files."""
        with patch("backend.llm.key_utils._dotenv_key_values", return_value={}):
            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": '"sk-test-quoted-key"', "OPEN_API_KEY": ""},
                clear=False,
            ):
                self.assertEqual(resolve_openai_api_key(), "sk-test-quoted-key")
                self.assertEqual(openai_key_source(), "OPENAI_API_KEY")

    def test_openai_chat_client_enabled_with_alias_key(self) -> None:
        """OpenAI chat client should initialize when only alias key is present."""
        with patch("backend.llm.key_utils._dotenv_key_values", return_value={}):
            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "", "OPEN_API_KEY": "sk-test-alias", "OPENAI_CHAT_MODEL": "gpt-4.1-mini"},
                clear=False,
            ):
                client = OpenAIChatClient()
                self.assertTrue(client.enabled)

    def test_openai_chat_client_refreshes_after_key_added(self) -> None:
        """Client should become enabled when key is added after initialization."""
        with patch("backend.llm.openai_client.OpenAI") as openai_ctor:
            with patch("backend.llm.key_utils._dotenv_key_values", return_value={}):
                with patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "", "OPEN_API_KEY": ""},
                    clear=False,
                ):
                    client = OpenAIChatClient()
                    self.assertFalse(client.enabled)
                    openai_ctor.assert_not_called()

                    os.environ["OPENAI_API_KEY"] = "sk-added-later"
                    self.assertTrue(client.enabled)
                    openai_ctor.assert_called_once()

    def test_dotenv_fallback_when_exported_env_is_empty(self) -> None:
        """Fallback to `.env` value when environment variables are present but blank."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "OPEN_API_KEY": ""},
            clear=False,
        ):
            with patch("backend.llm.key_utils._dotenv_key_values", return_value={"OPENAI_API_KEY": "sk-dotenv"}):
                self.assertEqual(resolve_openai_api_key(), "sk-dotenv")
                self.assertEqual(openai_key_source(), "OPENAI_API_KEY")

    def test_llm_health_reports_actionable_key_hint(self) -> None:
        """When no key exists, health status should explain exactly which vars are accepted."""
        with patch("backend.llm.key_utils._dotenv_key_values", return_value={}):
            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "", "OPEN_API_KEY": ""},
                clear=False,
            ):
                health = LlmHealthService().status()
                self.assertEqual(health.status, "Degraded")
                self.assertIsNotNone(health.error_summary)
                summary = str(health.error_summary)
                self.assertIn("OPENAI_API_KEY", summary)
                self.assertIn("OPEN_API_KEY", summary)


if __name__ == "__main__":
    unittest.main()
