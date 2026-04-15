from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time

from openai import AuthenticationError, APIConnectionError, OpenAI

from ..llm.key_utils import OPENAI_KEY_ENV_VARS, resolve_openai_api_key

# Probe result cached for this many seconds — avoids hammering the API on
# every UI render cycle while still reflecting key revocations within a minute.
_PROBE_TTL_SECONDS = 60


@dataclass(frozen=True)
class LlmHealthStatus:
    status: str
    checked_at: str
    error_summary: str | None


class LlmHealthService:
    """Reports LLM connectivity by probing the OpenAI API with a lightweight call."""

    def __init__(self) -> None:
        self._cached_status: LlmHealthStatus | None = None
        self._cached_key: str = ""
        self._cache_expires_at: float = 0.0

    def status(self) -> LlmHealthStatus:
        """Probes OpenAI with models.list(limit=1) to verify key is valid and active.

        Result is cached for ``_PROBE_TTL_SECONDS`` seconds per key value.
        Key presence alone is NOT sufficient — revoked keys return 401 and are
        reported as ``Degraded``.
        """
        api_key = resolve_openai_api_key()
        checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if not api_key:
            source_hint = ", ".join(OPENAI_KEY_ENV_VARS)
            return LlmHealthStatus(
                status="Degraded",
                checked_at=checked_at,
                error_summary=f"OpenAI key is not configured. Set one of: {source_hint}",
            )

        # Return cached result if same key and still within TTL.
        now = time.monotonic()
        if (
            self._cached_status is not None
            and api_key == self._cached_key
            and now < self._cache_expires_at
        ):
            return self._cached_status

        # Probe the API — models.list is the lightest authenticated endpoint.
        result = self._probe(api_key, checked_at)
        self._cached_status = result
        self._cached_key = api_key
        self._cache_expires_at = now + _PROBE_TTL_SECONDS
        return result

    @staticmethod
    def _probe(api_key: str, checked_at: str) -> LlmHealthStatus:
        """Makes a single authenticated API call to verify key validity."""
        try:
            OpenAI(api_key=api_key).models.list()
            return LlmHealthStatus(status="Connected", checked_at=checked_at, error_summary=None)
        except AuthenticationError as exc:
            return LlmHealthStatus(
                status="Degraded",
                checked_at=checked_at,
                error_summary=f"OpenAI key rejected (revoked or invalid): {exc.message}",
            )
        except APIConnectionError as exc:
            return LlmHealthStatus(
                status="Degraded",
                checked_at=checked_at,
                error_summary=f"Cannot reach OpenAI API: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return LlmHealthStatus(
                status="Degraded",
                checked_at=checked_at,
                error_summary=f"LLM health probe failed: {exc}",
            )
