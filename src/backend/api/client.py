from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from ..config import AppConfig, resolve_api_role
from ..services.agent_router_service import AgentRouterInput
from ..services.bootstrap_service import BootstrapArtifacts


class ApiClient:
    def __init__(self, config: AppConfig, artifacts: BootstrapArtifacts) -> None:
        """Initializes API client with remote base URL and local fallback artifacts."""
        self.config = config
        self.artifacts = artifacts
        self.base_url = config.api_base_url.rstrip("/")
        self.user_role = resolve_api_role(config, config.api_internal_token) or "viewer"

    @property
    def using_remote_api(self) -> bool:
        """Indicates whether calls should go over HTTP (`API_BASE_URL` set)."""
        return bool(self.base_url)

    def agent_router(
        self,
        *,
        session_id: str,
        query: str,
        language: str,
        filters: dict[str, str | None],
        mode: str,
        llm_provider: str = "chatgpt",
        llm_model: str | None = None,
        inference: dict[str, int | float] | None = None,
        input_method: str = "document",
        conversation_context: list[dict[str, str]] | None = None,
        applied_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Calls `/api/agent-router` remotely or local router fallback."""
        payload = {
            "session_id": session_id,
            "query": query,
            "language": language,
            "filters": filters,
            "mode": mode,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "inference": inference or {},
            "input_method": input_method,
            "conversation_context": conversation_context or [],
            "applied_context": applied_context or {},
        }
        return self._remote_or_fallback(
            lambda: self._post_json("/api/agent-router", payload),
            lambda: self._local_agent_router(
                session_id=session_id,
                query=query,
                language=language,
                filters=filters,
                mode=mode,
                llm_provider=llm_provider,
                llm_model=llm_model,
                inference=inference or {},
                input_method=input_method,
                conversation_context=conversation_context or [],
                applied_context=applied_context or {},
            ),
        )

    def auth_me(self) -> dict[str, Any]:
        """Returns resolved role for current token."""
        return self._remote_or_fallback(
            lambda: self._get_json("/api/auth/me"),
            lambda: {"role": self.user_role},
        )

    def compliance_status(self) -> dict[str, Any]:
        """Retrieves compliance status remotely or locally."""
        return self._remote_or_fallback(
            lambda: self._get_json("/api/compliance-status"),
            self._local_compliance_status,
        )

    def consent_record(self, *, session_id: str, user_consent: bool, locale: str) -> dict[str, Any]:
        """Records consent remotely or locally with current UTC timestamp."""
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {
            "session_id": session_id,
            "user_consent": user_consent,
            "timestamp": timestamp,
            "locale": locale,
        }
        return self._remote_or_fallback(
            lambda: self._post_json("/api/consent-record", payload),
            lambda: self._local_consent_record(
                session_id=session_id,
                user_consent=user_consent,
                locale=locale,
                timestamp=timestamp,
            ),
        )

    def llm_health(self) -> dict[str, Any]:
        """Retrieves LLM health remotely or locally."""
        return self._remote_or_fallback(
            lambda: self._get_json("/api/health/llm", include_auth=False),
            self._local_llm_health,
        )

    def metrics_sla(self) -> dict[str, Any]:
        """Retrieves SLA metrics remotely or from local observability store."""
        return self._remote_or_fallback(
            lambda: self._get_json("/api/metrics/sla"),
            lambda: self.artifacts.observability_store.sla_metrics(),
        )

    def monthly_snapshots(self) -> dict[str, Any]:
        """Retrieves monthly snapshots payload remotely or locally."""
        return self._remote_or_fallback(
            lambda: self._get_json("/api/metrics/monthly-snapshots"),
            self._local_monthly_snapshots,
        )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Retrieves trace payload remotely or locally."""
        return self._remote_or_fallback(
            lambda: self._get_json(f"/api/traces/{trace_id}"),
            lambda: self.artifacts.observability_store.get_trace(trace_id),
        )

    def governance_items(self, status: str | None = None, limit: int = 200) -> dict[str, Any]:
        """Retrieves governance listing remotely or locally."""
        return self._remote_or_fallback(
            lambda: self._get_json(
                "/api/governance-items",
                params={"status": status, "limit": limit} if status else {"limit": limit},
            ),
            lambda: {
                "items": self.artifacts.observability_store.list_governance_items(status=status, limit=limit),
                "counts": self.artifacts.observability_store.governance_counts(),
            },
        )

    def update_governance_status(self, item_id: int, status: str) -> dict[str, Any]:
        """Updates governance item status remotely or locally."""
        return self._remote_or_fallback(
            lambda: self._post_json(f"/api/governance-items/{item_id}/status", None, params={"status": status}),
            lambda: self._local_update_governance_status(item_id=item_id, status=status),
        )

    def question_library(self) -> dict[str, Any]:
        """Retrieves question-library payload remotely or locally."""
        return self._remote_or_fallback(
            lambda: self._get_json("/api/question-library"),
            self._local_question_library,
        )

    def index_health(self) -> dict[str, Any]:
        """Retrieves index-health diagnostics remotely or locally."""
        return self._remote_or_fallback(
            lambda: self._get_json("/api/index/health"),
            self._local_index_health,
        )

    def rebuild_index(self, *, refresh_prompts: bool = False) -> dict[str, Any]:
        """Requests reindex remotely or rebuilds local runtime artifacts."""
        return self._remote_or_fallback(
            lambda: self._post_json("/api/index/rebuild", {"refresh_prompts": refresh_prompts}),
            lambda: self._local_rebuild_index(refresh_prompts=refresh_prompts),
        )

    def submit_feedback(
        self,
        *,
        session_id: str,
        trace_id: str,
        score: int,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Submits thumbs-style feedback for an assistant response."""
        payload = {
            "session_id": session_id,
            "trace_id": trace_id,
            "score": int(score),
            "note": note,
        }
        return self._remote_or_fallback(
            lambda: self._post_json("/api/feedback", payload),
            lambda: {
                "feedback_id": self.artifacts.observability_store.record_feedback(
                    trace_id=trace_id,
                    session_id=session_id,
                    user_role=self.user_role,
                    score=int(score),
                    note=note,
                ),
                "stored": True,
            },
        )

    def unanswered_analytics(self, *, limit: int = 25) -> dict[str, Any]:
        """Returns unanswered-query analytics payload."""
        return self._remote_or_fallback(
            lambda: self._get_json("/api/analytics/unanswered", params={"limit": limit}),
            lambda: self.artifacts.observability_store.unanswered_analytics(limit=limit),
        )

    def save_search(
        self,
        *,
        session_id: str,
        query: str,
        filters: dict[str, str | None],
        applied_context: dict[str, str | None],
        pinned: bool = False,
    ) -> dict[str, Any]:
        """Stores query/filter snapshot for reopen workflows."""
        payload = {
            "session_id": session_id,
            "query": query,
            "filters": filters,
            "applied_context": applied_context,
            "pinned": bool(pinned),
        }
        return self._remote_or_fallback(
            lambda: self._post_json("/api/searches", payload),
            lambda: {
                "search_id": self.artifacts.observability_store.save_search(
                    session_id=session_id,
                    user_role=self.user_role,
                    query_text=query,
                    filters=filters,
                    applied_context=applied_context,
                    pinned=bool(pinned),
                ),
                "stored": True,
            },
        )

    def list_searches(
        self,
        *,
        limit: int = 25,
        session_id: str | None = None,
        pinned_only: bool = False,
    ) -> dict[str, Any]:
        """Returns recent saved searches."""
        params: dict[str, object] = {"limit": limit, "pinned_only": pinned_only}
        if session_id:
            params["session_id"] = session_id
        return self._remote_or_fallback(
            lambda: self._get_json("/api/searches", params=params),
            lambda: {
                "items": self.artifacts.observability_store.list_saved_searches(
                    limit=limit,
                    session_id=session_id,
                    user_role=self.user_role,
                    pinned_only=pinned_only,
                )
            },
        )

    def reopen_search(self, *, search_id: int) -> dict[str, Any]:
        """Marks a saved search as reopened and returns entry payload."""
        return self._remote_or_fallback(
            lambda: self._post_json(f"/api/searches/{search_id}/reopen", {}),
            lambda: {"item": self.artifacts.observability_store.reopen_saved_search(search_id=search_id)},
        )

    def _local_agent_router(
        self,
        *,
        session_id: str,
        query: str,
        language: str,
        filters: dict[str, str | None],
        mode: str,
        llm_provider: str,
        llm_model: str | None,
        inference: dict[str, int | float],
        input_method: str,
        conversation_context: list[dict[str, str]],
        applied_context: dict[str, str],
    ) -> dict[str, Any]:
        """Local adapter call into `AgentRouterService.handle`."""
        output = self.artifacts.agent_router_service.handle(
            AgentRouterInput(
                session_id=session_id,
                query=query,
                language=language,
                filters=filters,
                mode=mode,
                llm_provider=llm_provider,
                llm_model=llm_model,
                inference=inference,
                input_method=input_method,
                conversation_context=conversation_context,
                applied_context=applied_context,
                user_role=self.user_role,
            )
        )
        return {
            "trace_id": output.trace_id,
            "route_used": output.route_used,
            "response": output.response,
            "language_out": output.language_out,
            "fallback_used": output.fallback_used,
            "retrieval_mode": output.retrieval_mode,
            "confidence_score": output.confidence_score,
            "cache_status": output.cache_status,
            "labels": output.labels,
            "takeaways": output.takeaways,
            "latency_ms": output.latency_ms,
            "cards": [asdict(card) for card in output.cards],
            "sources": [asdict(source) for source in output.sources],
            "answer_mode": output.answer_mode,
            "needs_clarification": output.needs_clarification,
            "unanswered_reason": output.unanswered_reason,
            "fallback_reason": output.fallback_reason,
            "follow_up_suggestion": output.follow_up_suggestion,
            "applied_context": output.applied_context,
            "suggestions": output.suggestions,
        }

    def _local_compliance_status(self) -> dict[str, Any]:
        """Local compliance-status builder."""
        health = self.artifacts.agent_router_service.llm_health_service.status()
        return {
            "consent_enforced": True,
            "safety_policy_version": self.config.governance_policy_version,
            "governance_counts": self.artifacts.observability_store.governance_counts(),
            "system_health": {
                "llm_status": health.status,
                "last_check": health.checked_at,
                "error_summary": health.error_summary,
            },
        }

    def _local_consent_record(
        self,
        *,
        session_id: str,
        user_consent: bool,
        locale: str,
        timestamp: str,
    ) -> dict[str, Any]:
        """Local consent/session persistence helper."""
        self.artifacts.observability_store.record_session_start(session_id, locale=locale, consent=user_consent)
        level = "full" if user_consent else "minimal"
        record_id = self.artifacts.observability_store.record_consent(
            session_id=session_id,
            user_consent=user_consent,
            timestamp=timestamp,
            locale=locale,
            effective_logging_level=level,
        )
        return {"record_id": record_id, "effective_logging_level": level}

    def _local_llm_health(self) -> dict[str, Any]:
        """Local LLM health payload helper."""
        health = self.artifacts.agent_router_service.llm_health_service.status()
        return {
            "status": health.status,
            "last_check_time": health.checked_at,
            "error_summary": health.error_summary,
        }

    def _local_monthly_snapshots(self) -> dict[str, Any]:
        """Local monthly snapshot generation helper."""
        self.artifacts.observability_store.create_monthly_snapshot()
        return {
            "snapshots": self.artifacts.observability_store.monthly_snapshots(),
            "session_rollups": self.artifacts.observability_store.session_rollups(limit=100),
        }

    def _local_update_governance_status(self, *, item_id: int, status: str) -> dict[str, Any]:
        """Local governance status update helper."""
        self.artifacts.agent_router_service.governance_service.set_status(item_id, status)
        return {"ok": True, "item_id": item_id, "status": status}

    def _local_question_library(self) -> dict[str, Any]:
        """Local question-library payload helper."""
        return {
            "top_questions": self.artifacts.top_questions,
            "generated_questions": self.artifacts.generated_questions,
            "approved_library": self.artifacts.observability_store.list_governance_items(status="Approved", limit=200),
        }

    def _local_index_health(self) -> dict[str, Any]:
        """Local index/cache diagnostics payload helper."""
        semantic_cache_stats = self.artifacts.chatbot_service.retriever.semantic.embedding_cache_stats()
        answer_cache_stats = self.artifacts.chatbot_service.answer_cache_stats()
        router_cache_stats = self.artifacts.agent_router_service.query_cache.stats()

        def hit_rate(hits: int, misses: int) -> float:
            """Nested helper that computes cache hit-rate percentages safely."""
            total = hits + misses
            return round(hits / total, 4) if total else 0.0

        return {
            "index_version": self.artifacts.index_version,
            "signature": self.artifacts.index_signature,
            "document_count": self.artifacts.index_document_count,
            "chunk_count": self.artifacts.index_chunk_count,
            "last_rebuild_epoch": self.artifacts.index_last_rebuild_epoch,
            "embedding_mode": self.artifacts.index_embedding_mode,
            "cache_status_at_startup": self.artifacts.cache_status_at_startup,
            "cache_status_current": self.artifacts.cache_status.state,
            "cache_hit_rates": {
                "embedding": hit_rate(semantic_cache_stats.hits, semantic_cache_stats.misses),
                "answer": hit_rate(answer_cache_stats.hits, answer_cache_stats.misses),
                "router": hit_rate(router_cache_stats.hits, router_cache_stats.misses),
            },
            "cache_counters": {
                "embedding_hits": semantic_cache_stats.hits,
                "embedding_misses": semantic_cache_stats.misses,
                "answer_hits": answer_cache_stats.hits,
                "answer_misses": answer_cache_stats.misses,
                "router_hits": router_cache_stats.hits,
                "router_misses": router_cache_stats.misses,
            },
        }

    def _local_rebuild_index(self, *, refresh_prompts: bool) -> dict[str, Any]:
        """Rebuilds local runtime and returns fresh index-health payload."""
        self.artifacts = BootstrapService(self.config).build(
            force_rebuild_cache=True,
            force_refresh_prompts=refresh_prompts,
        )
        return {"ok": True, "index_health": self._local_index_health()}

    def _remote_or_fallback(self, remote_fn: Callable[[], Any], fallback_fn: Callable[[], Any]) -> Any:
        """Executes remote path when available; falls back on HTTP error."""
        if not self.using_remote_api:
            return fallback_fn()
        try:
            return remote_fn()
        except httpx.HTTPError:
            return fallback_fn()

    def _get_json(
        self,
        path: str,
        *,
        params: dict[str, object] | None = None,
        include_auth: bool = True,
    ) -> dict[str, Any]:
        """Executes authenticated GET and returns JSON payload."""
        headers = self._headers(include_auth=include_auth)
        with httpx.Client(timeout=25.0) as client:
            response = client.get(f"{self.base_url}{path}", headers=headers, params=params)
            response.raise_for_status()
            return response.json()

    def _post_json(
        self,
        path: str,
        payload: dict[str, object] | None,
        *,
        params: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """Executes authenticated POST and returns JSON payload."""
        headers = self._headers(include_auth=True)
        with httpx.Client(timeout=40.0) as client:
            response = client.post(f"{self.base_url}{path}", headers=headers, json=payload, params=params)
            response.raise_for_status()
            return response.json()

    def _headers(self, *, include_auth: bool) -> dict[str, str]:
        """Builds request headers, including internal token when required."""
        headers = {"accept": "application/json"}
        if include_auth:
            headers["x-internal-token"] = self.config.api_internal_token
        return headers
