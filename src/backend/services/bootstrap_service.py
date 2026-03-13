from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from ..business_rules import RETRIEVAL_RULES
from ..cache.index_cache import CacheInspectResult, IndexCache, build_signature
from ..config import AppConfig
from ..llm.openai_client import OpenAIChatClient
from ..loaders.excel_repository import ExcelRepository
from ..loaders.survey_prompt_loader import SurveyPromptLoader
from ..retrieval.hybrid import HybridRetriever
from .chatbot_service import ChatbotService


@dataclass(frozen=True)
class BootstrapArtifacts:
    chatbot_service: ChatbotService
    surveys: list[str]
    waves: list[str]
    topics: list[str]
    topic_source_types: list[str]
    cache_status: CacheInspectResult
    cache_status_at_startup: str
    cache_rebuilt: bool
    starter_prompts: list[str]


class BootstrapService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build(
        self,
        force_rebuild_cache: bool = False,
        force_refresh_prompts: bool = False,
    ) -> BootstrapArtifacts:
        data_dir = _infer_data_dir(self.config.excel_source_path)
        source_paths = _collect_excel_sources(self.config.excel_source_path, data_dir)
        repository = ExcelRepository(source_paths)
        cache = IndexCache(self.config.index_cache_dir)

        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        has_openai_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
        rules_fingerprint = (
            f"{RETRIEVAL_RULES.lexical_weight}:"
            f"{RETRIEVAL_RULES.semantic_weight}:"
            f"{RETRIEVAL_RULES.top_k}:"
            f"{RETRIEVAL_RULES.min_score_threshold}"
        )

        signature = build_signature(
            excel_source_paths=source_paths,
            embedding_model=embedding_model,
            has_openai_key=has_openai_key,
            rules_fingerprint=rules_fingerprint,
        )

        startup_cache_status = cache.inspect(signature)
        if force_rebuild_cache:
            cache.clear()
        retriever = None if force_rebuild_cache else cache.load(signature)
        rebuilt = False
        if retriever is None:
            records = repository.load_records()
            retriever = HybridRetriever.build(records)
            cache.save(signature, retriever)
            rebuilt = True
        else:
            records = retriever.records

        prompt_loader = SurveyPromptLoader(data_dir=data_dir, cache_dir=self.config.index_cache_dir)
        if force_refresh_prompts:
            prompt_loader.clear_cache()
        starter_prompts = prompt_loader.load_prompts(max_prompts=24, force_refresh=force_refresh_prompts)

        service = ChatbotService(retriever=retriever, llm_client=OpenAIChatClient())
        surveys = sorted({r.survey_name for r in records if r.survey_name})
        waves = sorted({r.wave_year for r in records if r.wave_year})
        topics = sorted({topic for r in records for topic in r.topic_labels})
        topic_source_types = sorted({source for r in records for source in r.topic_label_sources.values()})
        cache_status = cache.inspect(signature)

        return BootstrapArtifacts(
            chatbot_service=service,
            surveys=surveys,
            waves=waves,
            topics=topics,
            topic_source_types=topic_source_types,
            cache_status=cache_status,
            cache_status_at_startup=startup_cache_status.state,
            cache_rebuilt=rebuilt,
            starter_prompts=starter_prompts,
        )


def _infer_data_dir(excel_source_path: Path) -> Path:
    if excel_source_path.parent.exists():
        return excel_source_path.parent
    return Path.cwd() / "data"


def _collect_excel_sources(primary_source: Path, data_dir: Path) -> list[Path]:
    sources: list[Path] = []
    if primary_source.exists():
        sources.append(primary_source)
    uploads_dir = data_dir / "user_uploads"
    if uploads_dir.exists():
        for xlsx_file in sorted(uploads_dir.rglob("*.xlsx")):
            if xlsx_file not in sources:
                sources.append(xlsx_file)
    return sources
