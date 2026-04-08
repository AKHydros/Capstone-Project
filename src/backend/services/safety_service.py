from __future__ import annotations

from dataclasses import dataclass
import re


_BLOCKED_TOPICS = (
    "build a bomb",
    "credit card number",
    "steal password",
    "social insurance number",
    "social security number",
)


@dataclass(frozen=True)
class SafetyResult:
    allowed: bool
    reason: str


class SafetyService:
    def check_user_query(self, query: str) -> SafetyResult:
        lowered = query.lower()
        for phrase in _BLOCKED_TOPICS:
            if phrase in lowered:
                return SafetyResult(allowed=False, reason="blocked_topic")
        return SafetyResult(allowed=True, reason="ok")

    def check_assistant_response(self, response_text: str) -> SafetyResult:
        lowered = response_text.lower()
        for phrase in _BLOCKED_TOPICS:
            if phrase in lowered:
                return SafetyResult(allowed=False, reason="unsafe_response")
        return SafetyResult(allowed=True, reason="ok")

    def redact_pii(self, text: str) -> str:
        redacted = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
        redacted = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[redacted-phone]", redacted)
        return redacted
