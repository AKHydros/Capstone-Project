from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    excel_source_path: Path
    index_cache_dir: Path
    observability_db_path: Path
    logs_dir: Path
    api_internal_token: str
    api_base_url: str
    governance_policy_version: str
    default_router_mode: str
    openai_chat_model: str
    openai_chat_models: tuple[str, ...]
    rag_enabled: bool
    rag_confidence_threshold: float
    rag_score_gap_threshold: float
    rag_embed_cache_ttl: int
    rag_answer_cache_ttl: int
    rag_batch_rebuild: bool


def _parse_csv_env(raw: str) -> tuple[str, ...]:
    """Parses comma-separated env values into unique ordered tuple."""
    seen: set[str] = set()
    values: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return tuple(values)


def _parse_bool_env(raw: str | None, *, default: bool) -> bool:
    """Parses boolean-like env strings with default fallback."""
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Loads all runtime configuration from environment variables (memoized)."""
    source = os.getenv(
        "EXCEL_SOURCE_PATH",
        "data/market_research_capstone_draft_1.xlsx",
    )
    cache_dir = os.getenv(
        "INDEX_CACHE_DIR",
        "data/cache",
    )
    observability_db = os.getenv(
        "OBSERVABILITY_DB_PATH",
        "data/observability.db",
    )
    logs_dir = os.getenv(
        "LOGS_DIR",
        "data/logs",
    )
    api_internal_token = os.getenv("API_INTERNAL_TOKEN", "dev-internal-token")
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    governance_policy_version = os.getenv("GOVERNANCE_POLICY_VERSION", "v1")
    default_router_mode = os.getenv("DEFAULT_ROUTER_MODE", "hybrid").strip().lower()
    openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    openai_chat_models_raw = os.getenv("OPENAI_CHAT_MODELS", "").strip()
    openai_chat_models = _parse_csv_env(openai_chat_models_raw) if openai_chat_models_raw else (openai_chat_model,)
    if openai_chat_model not in openai_chat_models:
        openai_chat_models = (openai_chat_model, *openai_chat_models)
    rag_enabled = _parse_bool_env(os.getenv("RAG_ENABLED"), default=True)
    rag_batch_rebuild = _parse_bool_env(os.getenv("RAG_BATCH_REBUILD"), default=True)
    rag_confidence_threshold = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.72") or "0.72")
    rag_score_gap_threshold = float(os.getenv("RAG_SCORE_GAP_THRESHOLD", "0.07") or "0.07")
    rag_embed_cache_ttl = int(os.getenv("RAG_EMBED_CACHE_TTL", "1800") or "1800")
    rag_answer_cache_ttl = int(os.getenv("RAG_ANSWER_CACHE_TTL", "600") or "600")

    return AppConfig(
        excel_source_path=Path(source),
        index_cache_dir=Path(cache_dir),
        observability_db_path=Path(observability_db),
        logs_dir=Path(logs_dir),
        api_internal_token=api_internal_token,
        api_base_url=api_base_url,
        governance_policy_version=governance_policy_version,
        default_router_mode=default_router_mode,
        openai_chat_model=openai_chat_model,
        openai_chat_models=openai_chat_models,
        rag_enabled=rag_enabled,
        rag_confidence_threshold=rag_confidence_threshold,
        rag_score_gap_threshold=rag_score_gap_threshold,
        rag_embed_cache_ttl=rag_embed_cache_ttl,
        rag_answer_cache_ttl=rag_answer_cache_ttl,
        rag_batch_rebuild=rag_batch_rebuild,
    )


def reset_config_cache() -> None:
    """Clears memoized config to force re-read from environment."""
    load_config.cache_clear()
