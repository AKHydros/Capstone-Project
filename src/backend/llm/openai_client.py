from __future__ import annotations

import os

from openai import OpenAI


class OpenAIChatClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=api_key) if api_key else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def summarize(self, system_prompt: str, user_prompt: str) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client is not configured")

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.output_text
