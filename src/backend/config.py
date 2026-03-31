from __future__ import annotations

import os
from dataclasses import dataclass
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


def _parse_csv_env(raw: str) -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return tuple(values)



def load_config() -> AppConfig:
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
    )
