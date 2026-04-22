from __future__ import annotations

import os

from dotenv import dotenv_values, find_dotenv


OPENAI_KEY_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY", "OPEN_API_KEY")


def _dotenv_key_values() -> dict[str, str]:
    """Load key-related values from `.env` without mutating process env."""
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        return {}
    raw = dotenv_values(dotenv_path)
    values: dict[str, str] = {}
    for name in OPENAI_KEY_ENV_VARS:
        value = raw.get(name)
        if value is None:
            continue
        values[name] = str(value)
    return values


def _normalize_secret(raw: str) -> str:
    """Trim whitespace and optional surrounding quotes from env secrets."""
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def resolve_openai_api_key() -> str:
    """Return the first configured OpenAI key from supported env vars.

    Reads ``.env`` on every call so key rotations take effect without a
    process restart.  ``.env`` values take precedence over the stale
    process-environment copy that ``load_dotenv()`` (called once at import
    in ``config.py``) may have baked in at startup.
    """
    dotenv_values_map = _dotenv_key_values()
    for env_name in OPENAI_KEY_ENV_VARS:
        # Prefer the live .env file value over the stale os.environ snapshot.
        value = _normalize_secret(dotenv_values_map.get(env_name, ""))
        if not value:
            value = _normalize_secret(os.getenv(env_name, ""))
        if value:
            # Sync os.environ so other callers (e.g. OpenAI SDK) see the
            # current value without needing a restart.
            os.environ[env_name] = value
            return value
    return ""


def openai_key_source() -> str | None:
    """Return the env var name providing the active OpenAI key, if any."""
    dotenv_values_map = _dotenv_key_values()
    for env_name in OPENAI_KEY_ENV_VARS:
        value = _normalize_secret(os.getenv(env_name, ""))
        if not value:
            value = _normalize_secret(dotenv_values_map.get(env_name, ""))
        if value:
            return env_name
    return None
