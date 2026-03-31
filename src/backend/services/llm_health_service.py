from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os


@dataclass(frozen=True)
class LlmHealthStatus:
    status: str
    checked_at: str
    error_summary: str | None


class LlmHealthService:
    def status(self) -> LlmHealthStatus:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if api_key:
            return LlmHealthStatus(status="Connected", checked_at=checked_at, error_summary=None)
        return LlmHealthStatus(
            status="Degraded",
            checked_at=checked_at,
            error_summary="OPENAI_API_KEY is not configured",
        )
