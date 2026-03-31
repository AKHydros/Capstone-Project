from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from ..config import AppConfig
from ..services.agent_router_service import AgentRouterInput
from ..services.bootstrap_service import BootstrapArtifacts


class ApiClient:
    def __init__(self, config: AppConfig, artifacts: BootstrapArtifacts) -> None:
        self.config = config
        self.artifacts = artifacts
        self.base_url = config.api_base_url.rstrip("/")

    @property
    def using_remote_api(self) -> bool:
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
    ) -> dict[str, Any]:
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
            ),
        )

    def compliance_status(self) -> dict[str, Any]:
        return self._remote_or_fallback(
            lambda: self._get_json("/api/compliance-status"),
            self._local_compliance_status,
        )

    def consent_record(self, *, session_id: str, user_consent: bool, locale: str) -> dict[str, Any]:
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
        return self._remote_or_fallback(
            lambda: self._get_json("/api/health/llm", include_auth=False),
            self._local_llm_health,
        )

    def metrics_sla(self) -> dict[str, Any]:
        return self._remote_or_fallback(
            lambda: self._get_json("/api/metrics/sla"),
            lambda: self.artifacts.observability_store.sla_metrics(),
        )

    def monthly_snapshots(self) -> dict[str, Any]:
        return self._remote_or_fallback(
            lambda: self._get_json("/api/metrics/monthly-snapshots"),
            self._local_monthly_snapshots,
        )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return self._remote_or_fallback(
            lambda: self._get_json(f"/api/traces/{trace_id}"),
            lambda: self.artifacts.observability_store.get_trace(trace_id),
        )

    def governance_items(self, status: str | None = None, limit: int = 200) -> dict[str, Any]:
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
        return self._remote_or_fallback(
            lambda: self._post_json(f"/api/governance-items/{item_id}/status", None, params={"status": status}),
            lambda: self._local_update_governance_status(item_id=item_id, status=status),
        )

    def question_library(self) -> dict[str, Any]:
        return self._remote_or_fallback(
            lambda: self._get_json("/api/question-library"),
            self._local_question_library,
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
    ) -> dict[str, Any]:
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
            )
        )
        return {
            "trace_id": output.trace_id,
            "route_used": output.route_used,
            "response": output.response,
            "language_out": output.language_out,
            "fallback_used": output.fallback_used,
            "labels": output.labels,
            "takeaways": output.takeaways,
            "latency_ms": output.latency_ms,
            "cards": [asdict(card) for card in output.cards],
        }

    def _local_compliance_status(self) -> dict[str, Any]:
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
        health = self.artifacts.agent_router_service.llm_health_service.status()
        return {
            "status": health.status,
            "last_check_time": health.checked_at,
            "error_summary": health.error_summary,
        }

    def _local_monthly_snapshots(self) -> dict[str, Any]:
        self.artifacts.observability_store.create_monthly_snapshot()
        return {
            "snapshots": self.artifacts.observability_store.monthly_snapshots(),
            "session_rollups": self.artifacts.observability_store.session_rollups(limit=100),
        }

    def _local_update_governance_status(self, *, item_id: int, status: str) -> dict[str, Any]:
        self.artifacts.agent_router_service.governance_service.set_status(item_id, status)
        return {"ok": True, "item_id": item_id, "status": status}

    def _local_question_library(self) -> dict[str, Any]:
        return {
            "top_questions": self.artifacts.top_questions,
            "generated_questions": self.artifacts.generated_questions,
            "approved_library": self.artifacts.observability_store.list_governance_items(status="Approved", limit=200),
        }

    def _remote_or_fallback(self, remote_fn: Callable[[], Any], fallback_fn: Callable[[], Any]) -> Any:
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
        headers = self._headers(include_auth=True)
        with httpx.Client(timeout=40.0) as client:
            response = client.post(f"{self.base_url}{path}", headers=headers, json=payload, params=params)
            response.raise_for_status()
            return response.json()

    def _headers(self, *, include_auth: bool) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if include_auth:
            headers["x-internal-token"] = self.config.api_internal_token
        return headers
