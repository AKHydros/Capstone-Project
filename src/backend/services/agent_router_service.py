from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Any
from uuid import uuid4

from ..cache.ttl_cache import TTLCache
from ..models import QuestionRecord
from ..observability.db import ObservabilityStore
from ..observability.json_logger import JsonEventLogger
from .chatbot_service import ChatbotService
from .governance_service import GovernanceService
from .llm_health_service import LlmHealthService
from .safety_service import SafetyService
from .translation_service import TranslationService


VALID_MODES = {"hybrid", "llm", "deterministic"}


@dataclass(frozen=True)
class AgentRouterInput:
    session_id: str
    query: str
    language: str = "en"
    filters: dict[str, str | None] | None = None
    mode: str = "hybrid"
    trace_id: str | None = None
    llm_provider: str = "chatgpt"
    llm_model: str | None = None
    inference: dict[str, int | float] | None = None
    input_method: str = "document"
    conversation_context: list[dict[str, str]] | None = None
    user_role: str = "viewer"


@dataclass(frozen=True)
class ResultCard:
    question_id: str
    question_text: str
    survey_name: str
    wave_year: str
    topic_labels: list[str]
    topic_label_sources: dict[str, str]
    measurement_level: str
    source_file: str


@dataclass(frozen=True)
class SourceCitation:
    index: int
    marker: str
    label: str
    question_id: str
    survey_name: str
    wave_year: str
    question_text: str


@dataclass(frozen=True)
class AgentRouterOutput:
    trace_id: str
    route_used: str
    response: str
    language_out: str
    fallback_used: bool
    retrieval_mode: str
    confidence_score: float | None
    cache_status: dict[str, bool]
    labels: list[str]
    takeaways: list[str]
    latency_ms: float
    cards: list[ResultCard]
    sources: list[SourceCitation]
    answer_mode: str
    needs_clarification: bool
    unanswered_reason: str | None
    fallback_reason: str | None
    follow_up_suggestion: str | None = None


class AgentRouterService:
    def __init__(
        self,
        *,
        chatbot_service: ChatbotService,
        store: ObservabilityStore,
        logger: JsonEventLogger,
        governance_service: GovernanceService,
        translation_service: TranslationService,
        safety_service: SafetyService,
        llm_health_service: LlmHealthService,
        default_mode: str = "hybrid",
    ) -> None:
        """Wires chatbot, safety, translation, governance, observability, and router cache."""
        self.chatbot_service = chatbot_service
        self.store = store
        self.logger = logger
        self.governance_service = governance_service
        self.translation_service = translation_service
        self.safety_service = safety_service
        self.llm_health_service = llm_health_service
        self.default_mode = default_mode if default_mode in VALID_MODES else "hybrid"
        self.query_cache: TTLCache[dict[str, Any]] = TTLCache(ttl_seconds=300, max_size=2048)

    def handle(self, request: AgentRouterInput) -> AgentRouterOutput:
        """Main routing pipeline: consent, safety, cache, route decision, answer generation, telemetry, governance persistence."""
        started = time.perf_counter()
        trace_id = request.trace_id or str(uuid4())
        language = self.translation_service.normalize_language(request.language)
        filters = request.filters or {}
        mode = request.mode if request.mode in VALID_MODES else self.default_mode
        normalized_role = request.user_role if request.user_role in {"viewer", "analyst", "admin"} else "viewer"

        consent = self.store.ensure_session_and_get_consent(
            request.session_id,
            locale=language,
        )
        if not consent:
            response = "Please provide consent to continue. Until consent is provided, only minimal telemetry is stored."
            latency_ms = (time.perf_counter() - started) * 1000
            output = AgentRouterOutput(
                trace_id=trace_id,
                route_used="consent_gate",
                response=response,
                language_out=language,
                fallback_used=False,
                retrieval_mode="deterministic",
                confidence_score=None,
                cache_status={"embedding_cache_hit": False, "answer_cache_hit": False},
                labels=["Compliance"],
                takeaways=["Consent required before processing user content."],
                latency_ms=round(latency_ms, 2),
                cards=[],
                sources=[],
                answer_mode="direct_answer",
                needs_clarification=False,
                unanswered_reason="no_cards",
                fallback_reason=None,
            )
            self._record_event(
                request,
                output,
                payload={"consent_required": True, "user_role": normalized_role},
                include_content=False,
            )
            self.store.record_unanswered(
                trace_id=trace_id,
                session_id=request.session_id,
                user_role=normalized_role,
                reason="no_cards",
                query_text=self.safety_service.redact_pii(request.query),
            )
            return output

        translated_query = self.translation_service.translate(request.query, source_language=language, target_language="en")

        # --- Safety check (absolute violations) ---
        safety_result = self.safety_service.check_user_query(translated_query.text)
        if not safety_result.allowed:
            blocked_message = "I cannot assist with that request due to safety policy."
            translated = self.translation_service.translate(blocked_message, source_language="en", target_language=language)
            latency_ms = (time.perf_counter() - started) * 1000
            output = AgentRouterOutput(
                trace_id=trace_id,
                route_used="safety_block",
                response=translated.text,
                language_out=language,
                fallback_used=True,
                retrieval_mode="deterministic",
                confidence_score=None,
                cache_status={"embedding_cache_hit": False, "answer_cache_hit": False},
                labels=["Safety"],
                takeaways=["Blocked by deterministic safety policy."],
                latency_ms=round(latency_ms, 2),
                cards=[],
                sources=[],
                answer_mode="direct_answer",
                needs_clarification=False,
                unanswered_reason="no_cards",
                fallback_reason="safety_fallback",
            )
            self._record_event(
                request,
                output,
                payload={"reason": safety_result.reason, "user_role": normalized_role},
                include_content=True,
            )
            self.store.record_unanswered(
                trace_id=trace_id,
                session_id=request.session_id,
                user_role=normalized_role,
                reason="no_cards",
                query_text=self.safety_service.redact_pii(request.query),
            )
            return output

        # --- Domain constraint check ---
        # Rejects questions that are clearly outside the market-research
        # data dictionary scope before any retrieval is attempted.
        domain_result = self.safety_service.check_domain(translated_query.text)
        if not domain_result.allowed:
            out_of_domain_message = (
                "I can only answer questions about the PMG market research data dictionary — "
                "survey variables, question labels, coded values, topic mappings, and related metadata. "
                "Your question appears to be outside that scope. "
                "Please try asking about a specific survey variable, question ID, or research topic."
            )
            translated_ood = self.translation_service.translate(
                out_of_domain_message, source_language="en", target_language=language
            )
            latency_ms = (time.perf_counter() - started) * 1000
            output = AgentRouterOutput(
                trace_id=trace_id,
                route_used="domain_block",
                response=translated_ood.text,
                language_out=language,
                fallback_used=True,
                retrieval_mode="deterministic",
                confidence_score=None,
                cache_status={"embedding_cache_hit": False, "answer_cache_hit": False},
                labels=["Out-of-Domain"],
                takeaways=["Query rejected: outside market-research dictionary scope."],
                latency_ms=round(latency_ms, 2),
                cards=[],
                sources=[],
                answer_mode="direct_answer",
                needs_clarification=False,
                unanswered_reason="out_of_domain",
                fallback_reason="domain_constraint",
            )
            self._record_event(
                request,
                output,
                payload={"reason": domain_result.reason, "user_role": normalized_role},
                include_content=True,
            )
            self.store.record_unanswered(
                trace_id=trace_id,
                session_id=request.session_id,
                user_role=normalized_role,
                reason="out_of_domain",
                query_text=self.safety_service.redact_pii(request.query),
            )
            return output

        cache_key = self._cache_key(
            translated_query.text,
            filters,
            mode,
            language,
            request.llm_provider,
            request.llm_model,
            request.inference or {},
            request.conversation_context or [],
        )
        cached_payload = self.query_cache.get(cache_key)
        if cached_payload is not None:
            latency_ms = (time.perf_counter() - started) * 1000
            lookup_mode = str(cached_payload.get("lookup_mode", "hybrid_fallback"))
            try:
                variant_count = int(cached_payload.get("variant_count", 0))
            except (TypeError, ValueError):
                variant_count = 0
            output = AgentRouterOutput(
                trace_id=trace_id,
                route_used=str(cached_payload["route_used"]),
                response=str(cached_payload["response"]),
                language_out=language,
                fallback_used=bool(cached_payload["fallback_used"]),
                retrieval_mode=str(cached_payload.get("retrieval_mode", "deterministic")),
                confidence_score=float(cached_payload["confidence_score"])
                if cached_payload.get("confidence_score") is not None
                else None,
                cache_status={
                    "embedding_cache_hit": bool(cached_payload.get("embedding_cache_hit", False)),
                    "answer_cache_hit": bool(cached_payload.get("answer_cache_hit", False)),
                },
                labels=list(cached_payload["labels"]),
                takeaways=list(cached_payload["takeaways"]),
                latency_ms=round(latency_ms, 2),
                cards=[ResultCard(**card) for card in cached_payload["cards"]],
                sources=[SourceCitation(**source) for source in cached_payload.get("sources", [])],
                answer_mode=str(cached_payload.get("answer_mode", "direct_answer")),
                needs_clarification=bool(cached_payload.get("needs_clarification", False)),
                unanswered_reason=(
                    str(cached_payload["unanswered_reason"])
                    if cached_payload.get("unanswered_reason") is not None
                    else None
                ),
                fallback_reason=(
                    str(cached_payload["fallback_reason"])
                    if cached_payload.get("fallback_reason") is not None
                    else None
                ),
                follow_up_suggestion=(
                    str(cached_payload["follow_up_suggestion"])
                    if cached_payload.get("follow_up_suggestion")
                    else None
                ),
            )
            self.store.record_trace(
                trace_id=trace_id,
                session_id=request.session_id,
                route_used=output.route_used,
                fallback_used=output.fallback_used,
                latency_ms=output.latency_ms,
                status="ok",
            )
            self._record_event(
                request,
                output,
                payload={
                    "cache": "hit",
                    "lookup_mode": lookup_mode,
                    "variant_count": variant_count,
                    "user_role": normalized_role,
                },
                include_content=True,
            )
            self._record_qa(request, translated_query.text, output)
            if output.unanswered_reason:
                self.store.record_unanswered(
                    trace_id=trace_id,
                    session_id=request.session_id,
                    user_role=normalized_role,
                    reason=output.unanswered_reason,
                    query_text=self.safety_service.redact_pii(request.query),
                )
            return output

        route_used, fallback_used, fallback_reason = self._decide_route(mode)
        provider_supported = request.llm_provider.lower() in {"chatgpt", "openai"}
        if route_used == "llm" and not provider_supported:
            route_used = "deterministic"
            fallback_used = True
            fallback_reason = "unsupported_provider"
        survey_name = filters.get("survey_name")
        wave_year = filters.get("wave_year")
        topic_label = filters.get("topic_label")
        topic_source_type = filters.get("topic_source_type")

        response_payload = self.chatbot_service.chat(
            translated_query.text,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
            use_llm=(route_used == "llm"),
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            inference=request.inference,
            conversation_context=request.conversation_context,
        )
        response_text = response_payload.answer
        response_safety = self.safety_service.check_assistant_response(response_text)
        if not response_safety.allowed:
            response_text = "I cannot provide that response due to safety policy."
            route_used = "deterministic"
            fallback_used = True
            fallback_reason = "safety_fallback"

        governance = self.governance_service.generate_labels_takeaways(
            query=translated_query.text,
            answer=response_text,
            ranked_results=response_payload.ranked_results,
        )

        translated_answer = self.translation_service.translate(response_text, source_language="en", target_language=language)
        cards = [self._to_card(item) for item in response_payload.ranked_results]
        sources = self._build_sources(cards)
        unanswered_reason = self._classify_unanswered(cards, response_payload.needs_clarification)

        latency_ms = (time.perf_counter() - started) * 1000
        output = AgentRouterOutput(
            trace_id=trace_id,
            route_used=route_used,
            response=translated_answer.text,
            language_out=language,
            fallback_used=fallback_used,
            retrieval_mode=response_payload.retrieval_mode,
            confidence_score=response_payload.confidence_score,
            cache_status={
                "embedding_cache_hit": response_payload.embedding_cache_hit,
                "answer_cache_hit": response_payload.answer_cache_hit,
            },
            labels=governance.labels,
            takeaways=governance.takeaways,
            latency_ms=round(latency_ms, 2),
            cards=cards,
            sources=sources,
            answer_mode=response_payload.answer_mode,
            needs_clarification=response_payload.needs_clarification,
            unanswered_reason=unanswered_reason or None,
            fallback_reason=fallback_reason or None,
            follow_up_suggestion=response_payload.follow_up_suggestion,
        )

        self.query_cache.set(
            cache_key,
            {
                "route_used": output.route_used,
                "response": output.response,
                "fallback_used": output.fallback_used,
                "retrieval_mode": output.retrieval_mode,
                "confidence_score": output.confidence_score,
                "embedding_cache_hit": output.cache_status.get("embedding_cache_hit", False),
                "answer_cache_hit": output.cache_status.get("answer_cache_hit", False),
                "lookup_mode": response_payload.lookup_mode,
                "variant_count": response_payload.variant_count,
                "labels": output.labels,
                "takeaways": output.takeaways,
                "cards": [asdict(card) for card in output.cards],
                "sources": [asdict(source) for source in output.sources],
                "answer_mode": output.answer_mode,
                "needs_clarification": output.needs_clarification,
                "unanswered_reason": output.unanswered_reason,
                "fallback_reason": output.fallback_reason,
                "follow_up_suggestion": output.follow_up_suggestion,
            },
        )

        self.store.record_trace(
            trace_id=trace_id,
            session_id=request.session_id,
            route_used=output.route_used,
            fallback_used=output.fallback_used,
            latency_ms=output.latency_ms,
            status="ok",
        )
        self._record_event(
            request,
            output,
            payload={
                "cache": "miss",
                "lookup_mode": response_payload.lookup_mode,
                "variant_count": response_payload.variant_count,
                "user_role": normalized_role,
            },
            include_content=True,
        )
        self._record_qa(request, translated_query.text, output)
        if output.unanswered_reason:
            self.store.record_unanswered(
                trace_id=trace_id,
                session_id=request.session_id,
                user_role=normalized_role,
                reason=output.unanswered_reason,
                query_text=self.safety_service.redact_pii(request.query),
            )

        self.governance_service.save_qa_item(
            question_text=request.query,
            source="session_query",
            status="Draft",
            labels=output.labels,
            takeaways=output.takeaways,
            last_trace_id=trace_id,
        )

        return output

    def _record_qa(self, request: AgentRouterInput, query_en: str, output: AgentRouterOutput) -> None:
        """Persists redacted QA pair for traceability."""
        self.store.record_qa_pair(
            trace_id=output.trace_id,
            session_id=request.session_id,
            query_text=self.safety_service.redact_pii(query_en),
            response_text=self.safety_service.redact_pii(output.response),
            language_in=self.translation_service.normalize_language(request.language),
            language_out=output.language_out,
            labels=output.labels,
            takeaways=output.takeaways,
            route_used=output.route_used,
            fallback_used=output.fallback_used,
        )

    def _record_event(
        self,
        request: AgentRouterInput,
        output: AgentRouterOutput,
        payload: dict[str, Any],
        include_content: bool,
    ) -> None:
        """Writes structured event to DB and JSON logger.

        Empty strings are normalized to None so that analytics queries can
        reliably use IS NULL / IS NOT NULL rather than mixed empty-string checks.
        """
        event_payload: dict[str, Any] = {
            "mode": request.mode,
            "filters": request.filters or {},
            "llm_provider": request.llm_provider,
            "llm_model": request.llm_model or None,
            "inference": request.inference or {},
            "input_method": request.input_method,
            "retrieval_mode": output.retrieval_mode or None,
            # Always log the raw confidence score regardless of retrieval path so
            # deterministic responses are also observable in analytics.
            "confidence_score": output.confidence_score,
            "cache_status": output.cache_status,
            "answer_mode": output.answer_mode or None,
            "needs_clarification": output.needs_clarification,
            "unanswered_reason": output.unanswered_reason or None,
            "fallback_reason": output.fallback_reason or None,
            "user_role": request.user_role,
            **payload,
        }
        if include_content:
            event_payload.update(
                {
                    "query": self.safety_service.redact_pii(request.query),
                    "response": self.safety_service.redact_pii(output.response),
                    "labels": output.labels,
                }
            )

        self.store.record_event(
            session_id=request.session_id,
            trace_id=output.trace_id,
            event_type="agent_router",
            route_used=output.route_used,
            status="ok",
            latency_ms=output.latency_ms,
            payload=event_payload,
            error_text=None,
        )
        self.logger.log(
            {
                "session_id": request.session_id,
                "trace_id": output.trace_id,
                "event_type": "agent_router",
                "route_used": output.route_used,
                "status": "ok",
                "latency_ms": output.latency_ms,
                "payload": event_payload,
            }
        )

    def _cache_key(
        self,
        query_en: str,
        filters: dict[str, str | None],
        mode: str,
        language: str,
        llm_provider: str,
        llm_model: str | None,
        inference: dict[str, int | float],
        conversation_context: list[dict[str, str]],
    ) -> str:
        """Builds deterministic router-response cache key from request dimensions."""
        context_items = [
            {
                "role": str(turn.get("role", "")).strip().lower(),
                "content": str(turn.get("content", "")).strip(),
            }
            for turn in conversation_context[-20:]
        ]
        payload = {
            "q": query_en,
            "f": filters,
            "m": mode,
            "l": language,
            "p": llm_provider.lower(),
            "model": llm_model or "",
            "i": inference,
            "ctx": context_items,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _decide_route(self, mode: str) -> tuple[str, bool, str | None]:
        """Chooses deterministic vs LLM route based on mode and health."""
        health = self.llm_health_service.status()
        llm_available = self.chatbot_service.llm_client.enabled and health.status == "Connected"

        if mode == "deterministic":
            return "deterministic", False, None
        if mode == "llm":
            if llm_available:
                return "llm", False, None
            return "deterministic", True, "llm_unavailable"

        if llm_available:
            return "llm", False, None
        return "deterministic", True, "llm_unavailable"

    def _to_card(self, record: QuestionRecord) -> ResultCard:
        """Maps `QuestionRecord` to API response card shape."""
        return ResultCard(
            question_id=record.question_id,
            question_text=record.question_text,
            survey_name=record.survey_name,
            wave_year=record.wave_year,
            topic_labels=record.topic_labels,
            topic_label_sources=record.topic_label_sources,
            measurement_level=record.measurement_level,
            source_file=record.source_file,
        )

    def _build_sources(self, cards: list[ResultCard]) -> list[SourceCitation]:
        """Builds ordered citation markers from returned result cards."""
        sources: list[SourceCitation] = []
        for idx, card in enumerate(cards, start=1):
            sources.append(
                SourceCitation(
                    index=idx,
                    marker=f"[{idx}]",
                    label=f"{card.question_id} | {card.survey_name} {card.wave_year}",
                    question_id=card.question_id,
                    survey_name=card.survey_name,
                    wave_year=card.wave_year,
                    question_text=card.question_text,
                )
            )
        return sources

    def _classify_unanswered(self, cards: list[ResultCard], needs_clarification: bool) -> str | None:
        """Classifies unresolved interactions for unanswered-query analytics."""
        if not cards:
            return "no_cards"
        if needs_clarification:
            return "clarifier_only"
        return None
