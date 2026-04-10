from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..llm.key_utils import OPENAI_KEY_ENV_VARS, resolve_openai_api_key


@dataclass(frozen=True)
class LlmHealthStatus:
    status: str
    checked_at: str
    error_summary: str | None


class LlmHealthService:
    def status(self) -> LlmHealthStatus:
        """Reports `Connected` when API key exists, otherwise `Degraded` with reason."""
        api_key = resolve_openai_api_key()
        checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if api_key:
            return LlmHealthStatus(status="Connected", checked_at=checked_at, error_summary=None)
        source_hint = ", ".join(OPENAI_KEY_ENV_VARS)
        return LlmHealthStatus(
            status="Degraded",
            checked_at=checked_at,
            error_summary=f"OpenAI key is not configured. Set one of: {source_hint}",
        )
