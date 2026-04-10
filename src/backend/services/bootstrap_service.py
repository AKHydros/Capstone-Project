from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from ..business_rules import RETRIEVAL_RULES
from ..cache.index_cache import CacheInspectResult, IndexCache, build_signature
from ..config import AppConfig
from ..observability import JsonEventLogger, ObservabilityStore
from ..llm.openai_client import OpenAIChatClient
from ..llm.key_utils import resolve_openai_api_key
from ..loaders.excel_repository import ExcelRepository
from ..loaders.survey_prompt_loader import SurveyPromptLoader
from ..retrieval.hybrid import HybridRetriever
from .agent_router_service import AgentRouterService
from .chatbot_service import ChatbotService
from .governance_service import GovernanceService
from .llm_health_service import LlmHealthService
from .question_library_service import QuestionLibraryService
from .safety_service import SafetyService
from .translation_service import TranslationService


@dataclass(frozen=True)
class BootstrapArtifacts:
    chatbot_service: ChatbotService
    agent_router_service: AgentRouterService
    observability_store: ObservabilityStore
    doc_question_hints: dict[str, dict[str, list[str]]]
    surveys: list[str]
    waves: list[str]
    topics: list[str]
    topic_source_types: list[str]
    cache_status: CacheInspectResult
    cache_status_at_startup: str
    cache_rebuilt: bool
    index_version: str
    index_signature: str
    index_document_count: int
    index_chunk_count: int
    index_embedding_mode: str
    index_last_rebuild_epoch: int | None
    starter_prompts: list[str]
    top_questions: list[str]
    generated_questions: list[str]


class BootstrapService:
    def __init__(self, config: AppConfig) -> None:
        """Stores app configuration for runtime assembly."""
        self.config = config

    def build(
        self,
        force_rebuild_cache: bool = False,
        force_refresh_prompts: bool = False,
    ) -> BootstrapArtifacts:
        """Constructs full runtime graph: data loading, retriever, caches, services, governance, observability."""
        data_dir = _infer_data_dir(self.config.excel_source_path)
        source_paths = _collect_excel_sources(self.config.excel_source_path, data_dir)
        repository = ExcelRepository(source_paths)
        cache = IndexCache(self.config.index_cache_dir)

        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        has_openai_key = bool(resolve_openai_api_key())
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
            if self.config.rag_batch_rebuild or force_rebuild_cache:
                cache.save(signature, retriever)
                rebuilt = True
        else:
            records = retriever.records

        prompt_loader = SurveyPromptLoader(data_dir=data_dir, cache_dir=self.config.index_cache_dir)
        if force_refresh_prompts:
            prompt_loader.clear_cache()
        starter_prompts = prompt_loader.load_prompts(max_prompts=24, force_refresh=force_refresh_prompts)
        doc_question_hints = prompt_loader.load_question_hints(force_refresh=force_refresh_prompts)

        observability_store = ObservabilityStore(self.config.observability_db_path)
        event_logger = JsonEventLogger(self.config.logs_dir)
        llm_health_service = LlmHealthService()
        translation_service = TranslationService()
        safety_service = SafetyService()
        governance_service = GovernanceService(observability_store)
        question_library_service = QuestionLibraryService(governance_service)

        service = ChatbotService(
            retriever=retriever,
            llm_client=OpenAIChatClient(),
            doc_question_hints=doc_question_hints,
            rag_enabled_override=self.config.rag_enabled,
            rag_confidence_threshold_override=self.config.rag_confidence_threshold,
            rag_score_gap_threshold_override=self.config.rag_score_gap_threshold,
            rag_answer_cache_ttl_override=self.config.rag_answer_cache_ttl,
        )
        question_library = question_library_service.preload(records, iterations=10)
        agent_router_service = AgentRouterService(
            chatbot_service=service,
            store=observability_store,
            logger=event_logger,
            governance_service=governance_service,
            translation_service=translation_service,
            safety_service=safety_service,
            llm_health_service=llm_health_service,
            default_mode=self.config.default_router_mode,
        )
        surveys = sorted({r.survey_name for r in records if r.survey_name})
        waves = sorted({r.wave_year for r in records if r.wave_year})
        topics = sorted({topic for r in records for topic in r.topic_labels})
        topic_source_types = sorted({source for r in records for source in r.topic_label_sources.values()})
        cache_status = cache.inspect(signature)
        index_version = cache_status.index_version or "runtime"
        index_signature = cache_status.signature or signature
        index_document_count = cache_status.document_count or len(records)
        index_chunk_count = cache_status.chunk_count or len(getattr(retriever, "chunks", []))
        index_embedding_mode = cache_status.embedding_mode or retriever.semantic.mode
        index_last_rebuild_epoch = cache_status.created_at_epoch

        return BootstrapArtifacts(
            chatbot_service=service,
            agent_router_service=agent_router_service,
            observability_store=observability_store,
            doc_question_hints=doc_question_hints,
            surveys=surveys,
            waves=waves,
            topics=topics,
            topic_source_types=topic_source_types,
            cache_status=cache_status,
            cache_status_at_startup=startup_cache_status.state,
            cache_rebuilt=rebuilt,
            index_version=index_version,
            index_signature=index_signature,
            index_document_count=index_document_count,
            index_chunk_count=index_chunk_count,
            index_embedding_mode=index_embedding_mode,
            index_last_rebuild_epoch=index_last_rebuild_epoch,
            starter_prompts=starter_prompts,
            top_questions=question_library.top_questions,
            generated_questions=question_library.generated_questions,
        )


def _infer_data_dir(excel_source_path: Path) -> Path:
    """Determines data directory fallback when source parent is missing."""
    if excel_source_path.parent.exists():
        return excel_source_path.parent
    return Path.cwd() / "data"


def _collect_excel_sources(primary_source: Path, data_dir: Path) -> list[Path]:
    """Collects primary and uploaded Excel sources for ingestion."""
    sources: list[Path] = []
    if primary_source.exists():
        sources.append(primary_source)
    uploads_dir = data_dir / "user_uploads"
    if uploads_dir.exists():
        for xlsx_file in sorted(uploads_dir.rglob("*.xlsx")):
            if xlsx_file not in sources:
                sources.append(xlsx_file)
    return sources
