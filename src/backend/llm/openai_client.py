from __future__ import annotations

import os

from openai import OpenAI

from .key_utils import resolve_openai_api_key


class OpenAIChatClient:
    def __init__(self) -> None:
        """Initializes OpenAI client if key is configured and sets default model."""
        self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        self.client = None
        self._api_key = ""
        self._refresh_client_from_env()

    def _refresh_client_from_env(self) -> None:
        """Sync client instance with the latest configured key value."""
        api_key = resolve_openai_api_key()
        if not api_key:
            self.client = None
            self._api_key = ""
            return
        if self.client is not None and api_key == self._api_key:
            return
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=3)
        self._api_key = api_key

    @property
    def enabled(self) -> bool:
        """Indicates whether chat client is configured and available."""
        self._refresh_client_from_env()
        return self.client is not None

    def summarize(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
    ) -> str:
        """Sends grounded chat prompts to OpenAI Responses API and returns text."""
        self._refresh_client_from_env()
        if not self.client:
            raise RuntimeError("OpenAI client is not configured (set OPENAI_API_KEY or OPEN_API_KEY)")

        request_payload: dict[str, object] = {
            "model": model or self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1 if temperature is None else temperature,
        }
        if max_output_tokens is not None:
            request_payload["max_output_tokens"] = max_output_tokens
        if top_p is not None:
            request_payload["top_p"] = top_p

        response = self.client.responses.create(**request_payload)
        return response.output_text
