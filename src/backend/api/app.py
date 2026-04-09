from __future__ import annotations

from dataclasses import asdict
import time
import threading
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import load_config
from ..services.agent_router_service import AgentRouterInput
from ..services.bootstrap_service import BootstrapArtifacts, BootstrapService
from .schemas import (
    AgentRouterRequest,
    AgentRouterResponse,
    ComplianceStatusResponse,
    ConsentRecordRequest,
    ConsentRecordResponse,
    IndexHealthResponse,
    LlmHealthResponse,
    ReindexRequest,
    ReindexResponse,
)


_RUNTIME_LOCK = threading.Lock()
_RUNTIME_INSTANCE: BootstrapArtifacts | None = None


def _runtime() -> BootstrapArtifacts:
    """Lazily builds and returns a singleton `BootstrapArtifacts` runtime instance under a lock."""
    global _RUNTIME_INSTANCE
    with _RUNTIME_LOCK:
        if _RUNTIME_INSTANCE is None:
            config = load_config()
            _RUNTIME_INSTANCE = BootstrapService(config).build()
        return _RUNTIME_INSTANCE


def _rebuild_runtime(*, refresh_prompts: bool = False) -> BootstrapArtifacts:
    """Forces a full runtime rebuild (reindex + optional prompt refresh) and replaces the singleton instance."""
    global _RUNTIME_INSTANCE
    with _RUNTIME_LOCK:
        config = load_config()
        _RUNTIME_INSTANCE = BootstrapService(config).build(
            force_rebuild_cache=True,
            force_refresh_prompts=refresh_prompts,
        )
        return _RUNTIME_INSTANCE


def _auth_dependency(request: Request, x_internal_token: Annotated[str | None, Header()] = None) -> None:
    """Enforces internal token auth on API routes (except LLM health endpoint)."""
    if request.url.path == "/api/health/llm":
        return
    token = (x_internal_token or "").strip()
    config = load_config()
    if not token or token != config.api_internal_token:
        raise HTTPException(status_code=401, detail="Invalid internal token")


app = FastAPI(title="Capstone Chatbot API", version="1.0.0")


@app.on_event("startup")
def warm_runtime() -> None:
    """FastAPI startup hook that pre-warms runtime to avoid first-request cold-start latency."""
    _runtime()


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """Adds trace IDs, measures request latency, records API errors, and injects latency headers."""
    started = time.perf_counter()
    trace_id = request.headers.get("x-trace-id") or str(uuid4())
    request.state.trace_id = trace_id

    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        runtime = _runtime()
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        runtime.observability_store.record_event(
            session_id=request.headers.get("x-session-id", "unknown"),
            trace_id=trace_id,
            event_type="api_error",
            route_used=request.url.path,
            status="error",
            latency_ms=latency_ms,
            payload={"method": request.method},
            error_text=str(exc),
        )
        runtime.observability_store.record_trace(
            trace_id=trace_id,
            session_id=request.headers.get("x-session-id", "unknown"),
            route_used=request.url.path,
            fallback_used=False,
            latency_ms=latency_ms,
            status="error",
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "trace_id": trace_id})

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["x-trace-id"] = trace_id
    response.headers["x-latency-ms"] = str(latency_ms)
    return response


@app.get("/api/health/llm", response_model=LlmHealthResponse)
def get_llm_health() -> LlmHealthResponse:
    """Returns LLM connectivity status from `LlmHealthService`."""
    health = _runtime().agent_router_service.llm_health_service.status()
    return LlmHealthResponse(
        status=health.status,
        last_check_time=health.checked_at,
        error_summary=health.error_summary,
    )


@app.post("/api/consent-record", response_model=ConsentRecordResponse, dependencies=[Depends(_auth_dependency)])
def post_consent_record(payload: ConsentRecordRequest) -> ConsentRecordResponse:
    """Persists consent decisions and session logging level."""
    runtime = _runtime()
    runtime.observability_store.record_session_start(
        payload.session_id,
        locale=payload.locale,
        consent=payload.user_consent,
    )
    logging_level = "full" if payload.user_consent else "minimal"
    record_id = runtime.observability_store.record_consent(
        session_id=payload.session_id,
        user_consent=payload.user_consent,
        timestamp=payload.timestamp,
        locale=payload.locale,
        effective_logging_level=logging_level,
    )
    return ConsentRecordResponse(record_id=record_id, effective_logging_level=logging_level)


@app.get("/api/compliance-status", response_model=ComplianceStatusResponse, dependencies=[Depends(_auth_dependency)])
def get_compliance_status() -> ComplianceStatusResponse:
    """Returns consent enforcement, governance counts, and LLM health summary."""
    config = load_config()
    runtime = _runtime()
    health = runtime.agent_router_service.llm_health_service.status()
    return ComplianceStatusResponse(
        consent_enforced=True,
        safety_policy_version=config.governance_policy_version,
        governance_counts=runtime.observability_store.governance_counts(),
        system_health={
            "llm_status": health.status,
            "last_check": health.checked_at,
            "error_summary": health.error_summary,
        },
    )


@app.post("/api/agent-router", response_model=AgentRouterResponse, dependencies=[Depends(_auth_dependency)])
def post_agent_router(request: AgentRouterRequest, raw_request: Request) -> AgentRouterResponse:
    """Main chat route: converts API payload into router input and returns routed response payload."""
    runtime = _runtime()
    trace_id = request.trace_id or getattr(raw_request.state, "trace_id", str(uuid4()))
    output = runtime.agent_router_service.handle(
        AgentRouterInput(
            trace_id=trace_id,
            session_id=request.session_id,
            query=request.query,
            language=request.language,
            filters=request.filters.model_dump(),
            mode=request.mode,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            inference=request.inference.model_dump(exclude_none=True),
            input_method=request.input_method,
        )
    )
    return AgentRouterResponse(
        trace_id=output.trace_id,
        route_used=output.route_used,
        response=output.response,
        language_out=output.language_out,
        fallback_used=output.fallback_used,
        retrieval_mode=output.retrieval_mode,
        confidence_score=output.confidence_score,
        cache_status=output.cache_status,
        labels=output.labels,
        takeaways=output.takeaways,
        latency_ms=output.latency_ms,
        cards=[asdict(card) for card in output.cards],
    )


@app.get("/api/index/health", response_model=IndexHealthResponse, dependencies=[Depends(_auth_dependency)])
def get_index_health() -> IndexHealthResponse:
    """Returns index/cache diagnostics and cache hit-rate counters."""
    return _index_health_payload(_runtime())


@app.post("/api/index/rebuild", response_model=ReindexResponse, dependencies=[Depends(_auth_dependency)])
def post_index_rebuild(payload: ReindexRequest) -> ReindexResponse:
    """Triggers forced index rebuild and returns rebuilt index health."""
    runtime = _rebuild_runtime(refresh_prompts=payload.refresh_prompts)
    return ReindexResponse(ok=True, index_health=_index_health_payload(runtime))


@app.get("/api/metrics/sla", dependencies=[Depends(_auth_dependency)])
def get_metrics_sla() -> dict[str, float | int]:
    """Returns SLA metrics (latency percentiles, rates, counts)."""
    return _runtime().observability_store.sla_metrics()


@app.get("/api/metrics/monthly-snapshots", dependencies=[Depends(_auth_dependency)])
def get_metrics_monthly_snapshots() -> dict[str, object]:
    """Creates a monthly snapshot and returns snapshots + session rollups."""
    runtime = _runtime()
    runtime.observability_store.create_monthly_snapshot()
    return {
        "snapshots": runtime.observability_store.monthly_snapshots(),
        "session_rollups": runtime.observability_store.session_rollups(limit=100),
    }


@app.get("/api/traces/{trace_id}", dependencies=[Depends(_auth_dependency)])
def get_trace(trace_id: str) -> dict[str, object] | None:
    """Retrieves a trace plus latest QA pair for that trace."""
    return _runtime().observability_store.get_trace(trace_id)


@app.get("/api/governance-items", dependencies=[Depends(_auth_dependency)])
def list_governance_items(status: str | None = None, limit: int = 200) -> dict[str, object]:
    """Lists governance items with optional status filter and count summary."""
    runtime = _runtime()
    return {
        "items": runtime.observability_store.list_governance_items(status=status, limit=limit),
        "counts": runtime.observability_store.governance_counts(),
    }


@app.post("/api/governance-items/{item_id}/status", dependencies=[Depends(_auth_dependency)])
def update_governance_item_status(item_id: int, status: str) -> dict[str, object]:
    """Updates lifecycle status of a governance item."""
    runtime = _runtime()
    runtime.agent_router_service.governance_service.set_status(item_id, status)
    return {"ok": True, "item_id": item_id, "status": status}


@app.get("/api/question-library", dependencies=[Depends(_auth_dependency)])
def get_question_library() -> dict[str, object]:
    """Returns top questions, generated questions, and approved library items."""
    runtime = _runtime()
    return {
        "top_questions": runtime.top_questions,
        "generated_questions": runtime.generated_questions,
        "approved_library": runtime.observability_store.list_governance_items(status="Approved", limit=200),
    }


def _index_health_payload(runtime: BootstrapArtifacts) -> IndexHealthResponse:
    """Builds normalized index-health response from runtime caches and counters."""
    semantic_cache_stats = runtime.chatbot_service.retriever.semantic.embedding_cache_stats()
    answer_cache_stats = runtime.chatbot_service.answer_cache_stats()
    router_cache_stats = runtime.agent_router_service.query_cache.stats()

    def _hit_rate(hits: int, misses: int) -> float:
        """Nested helper that safely computes a percentage hit rate (0 when denominator is 0)."""
        total = hits + misses
        if total <= 0:
            return 0.0
        return round(hits / total, 4)

    cache_hit_rates = {
        "embedding": _hit_rate(semantic_cache_stats.hits, semantic_cache_stats.misses),
        "answer": _hit_rate(answer_cache_stats.hits, answer_cache_stats.misses),
        "router": _hit_rate(router_cache_stats.hits, router_cache_stats.misses),
    }
    cache_counters = {
        "embedding_hits": semantic_cache_stats.hits,
        "embedding_misses": semantic_cache_stats.misses,
        "answer_hits": answer_cache_stats.hits,
        "answer_misses": answer_cache_stats.misses,
        "router_hits": router_cache_stats.hits,
        "router_misses": router_cache_stats.misses,
    }

    return IndexHealthResponse(
        index_version=runtime.index_version,
        signature=runtime.index_signature,
        document_count=runtime.index_document_count,
        chunk_count=runtime.index_chunk_count,
        last_rebuild_epoch=runtime.index_last_rebuild_epoch,
        embedding_mode=runtime.index_embedding_mode,
        cache_status_at_startup=runtime.cache_status_at_startup,
        cache_status_current=runtime.cache_status.state,
        cache_hit_rates=cache_hit_rates,
        cache_counters=cache_counters,
    )
