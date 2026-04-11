from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
import time
import threading
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import load_config, resolve_api_role
from ..services.agent_router_service import AgentRouterInput
from ..services.bootstrap_service import BootstrapArtifacts, BootstrapService
from .schemas import (
    AgentRouterRequest,
    AgentRouterResponse,
    AuthMeResponse,
    ComplianceStatusResponse,
    ConsentRecordRequest,
    ConsentRecordResponse,
    FeedbackRequest,
    FeedbackResponse,
    IndexHealthResponse,
    LlmHealthResponse,
    ReopenSearchResponse,
    ReindexRequest,
    ReindexResponse,
    SaveSearchRequest,
    SaveSearchResponse,
    SavedSearchItemResponse,
    SavedSearchListResponse,
    UnansweredAnalyticsResponse,
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


def _auth_dependency(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_internal_token: Annotated[str | None, Header()] = None,
) -> str:
    """Enforces token auth and resolves role for protected API routes.

    Accepts tokens via two header schemes (in priority order):

    1. ``Authorization: Bearer <token>`` — standard RFC 6750 scheme.
    2. ``x-internal-token: <token>`` — legacy scheme kept for backward
       compatibility with existing tooling; will be removed in a future version.

    The ``/api/health/llm`` endpoint is intentionally public (no token required)
    so monitoring infrastructure can probe LLM connectivity without credentials.
    """
    if request.url.path == "/api/health/llm":
        request.state.user_role = "viewer"
        request.state.token_hash = ""
        return "viewer"

    # Prefer Authorization: Bearer, fall back to x-internal-token.
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[len("bearer "):].strip()
    elif x_internal_token:
        token = x_internal_token.strip()

    config = load_config()
    role = resolve_api_role(config, token)
    if role is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    request.state.user_role = role
    # Store a short hash of the token for session ownership checks (M5).
    # Not the full token — just enough to verify the same caller.
    request.state.token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
    return role


def _require_roles(*allowed_roles: str):
    """Creates dependency that enforces allowed roles."""
    allowed = set(allowed_roles)

    def _dependency(role: Annotated[str, Depends(_auth_dependency)]) -> str:
        if role not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient role for this endpoint")
        return role

    return _dependency


# ---------------------------------------------------------------------------
# Rate limiter — keyed by remote IP.
# Limits: 60 req/min on general routes, 30 req/min on the LLM-backed route.
# ---------------------------------------------------------------------------
_limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(title="Capstone Chatbot API", version="1.0.0")
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS — restrict cross-origin requests to known UI origins.
# Override CORS_ALLOWED_ORIGINS env var (comma-separated) for production.
# ---------------------------------------------------------------------------
_cors_origins = [
    o.strip()
    for o in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["authorization", "x-internal-token", "x-trace-id", "content-type"],
)


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects standard defensive HTTP response headers on every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-XSS-Protection", "1; mode=block")
        return response


app.add_middleware(_SecurityHeadersMiddleware)


@app.on_event("startup")
def warm_runtime() -> None:
    """FastAPI startup hook that pre-warms runtime and expires stale sessions (L3)."""
    rt = _runtime()
    expired = rt.observability_store.expire_stale_sessions(idle_hours=24)
    if expired:
        import logging as _logging
        _logging.getLogger(__name__).info("Expired %d stale session(s) on startup.", expired)


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
        # Only expose trace_id to analyst/admin roles — viewers get a generic error.
        user_role = getattr(request.state, "user_role", "viewer")
        body: dict[str, str] = {"detail": "Internal server error"}
        if user_role in {"analyst", "admin"}:
            body["trace_id"] = trace_id
        return JSONResponse(status_code=500, content=body)

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


@app.get(
    "/api/auth/me",
    response_model=AuthMeResponse,
    dependencies=[Depends(_require_roles("viewer", "analyst", "admin"))],
)
def get_auth_me(raw_request: Request) -> AuthMeResponse:
    """Returns resolved role for current token."""
    role = str(getattr(raw_request.state, "user_role", "viewer"))
    return AuthMeResponse(role=role)


@app.post(
    "/api/consent-record",
    response_model=ConsentRecordResponse,
    dependencies=[Depends(_require_roles("viewer", "analyst", "admin"))],
)
def post_consent_record(payload: ConsentRecordRequest, raw_request: Request) -> ConsentRecordResponse:
    """Persists consent decisions and session logging level.

    Enforces session ownership (M5): once a session has been registered by a
    token, only the **same** token (identified by its SHA-256 prefix) may
    modify its consent state.  A different token attempting to alter consent
    for an existing session receives a 403.
    """
    runtime = _runtime()
    token_hash: str = getattr(raw_request.state, "token_hash", "")

    # Check ownership of an existing session.
    existing_hash = runtime.observability_store.session_owner_hash(payload.session_id)
    if existing_hash is not None and token_hash and existing_hash != token_hash:
        raise HTTPException(
            status_code=403,
            detail="Consent may only be modified by the session owner.",
        )

    runtime.observability_store.record_session_start(
        payload.session_id,
        locale=payload.locale,
        consent=payload.user_consent,
        token_hash=token_hash,
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


@app.get(
    "/api/compliance-status",
    response_model=ComplianceStatusResponse,
    dependencies=[Depends(_require_roles("analyst", "admin"))],
)
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


@app.post(
    "/api/agent-router",
    response_model=AgentRouterResponse,
    dependencies=[Depends(_require_roles("viewer", "analyst", "admin"))],
)
@_limiter.limit("30/minute")
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
            conversation_context=[turn.model_dump() for turn in request.conversation_context],
            applied_context=request.applied_context,
            user_role=str(getattr(raw_request.state, "user_role", "viewer")),
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
        sources=[asdict(source) for source in output.sources],
        answer_mode=output.answer_mode,
        needs_clarification=output.needs_clarification,
        unanswered_reason=output.unanswered_reason,
        fallback_reason=output.fallback_reason,
        follow_up_suggestion=output.follow_up_suggestion,
        applied_context=output.applied_context,
        suggestions=output.suggestions,
    )


@app.post(
    "/api/searches",
    response_model=SaveSearchResponse,
    dependencies=[Depends(_require_roles("viewer", "analyst", "admin"))],
)
def post_saved_search(payload: SaveSearchRequest, raw_request: Request) -> SaveSearchResponse:
    """Persists a query + filter snapshot for quick reopen."""
    runtime = _runtime()
    role = str(getattr(raw_request.state, "user_role", "viewer"))
    search_id = runtime.observability_store.save_search(
        session_id=payload.session_id,
        user_role=role,
        query_text=payload.query,
        filters=payload.filters,
        applied_context=payload.applied_context,
        pinned=payload.pinned,
    )
    return SaveSearchResponse(search_id=search_id, stored=True)


@app.get(
    "/api/searches",
    response_model=SavedSearchListResponse,
    dependencies=[Depends(_require_roles("viewer", "analyst", "admin"))],
)
def get_saved_searches(
    raw_request: Request,
    limit: int = 25,
    session_id: str | None = None,
    user_role: str | None = None,
    pinned_only: bool = False,
) -> SavedSearchListResponse:
    """Returns recent saved searches for reopen workflows."""
    runtime = _runtime()
    role = str(getattr(raw_request.state, "user_role", "viewer"))
    effective_user_role = user_role if role == "admin" and user_role else role
    items = runtime.observability_store.list_saved_searches(
        limit=limit,
        session_id=session_id,
        user_role=effective_user_role,
        pinned_only=pinned_only,
    )
    return SavedSearchListResponse(items=[SavedSearchItemResponse(**item) for item in items])


@app.post(
    "/api/searches/{search_id}/reopen",
    response_model=ReopenSearchResponse,
    dependencies=[Depends(_require_roles("viewer", "analyst", "admin"))],
)
def post_reopen_search(search_id: int) -> ReopenSearchResponse:
    """Marks a saved search as reopened and returns the latest payload."""
    runtime = _runtime()
    item = runtime.observability_store.reopen_saved_search(search_id=search_id)
    if item is None:
        return ReopenSearchResponse(item=None)
    return ReopenSearchResponse(item=SavedSearchItemResponse(**item))


@app.get("/api/index/health", response_model=IndexHealthResponse, dependencies=[Depends(_require_roles("admin"))])
def get_index_health() -> IndexHealthResponse:
    """Returns index/cache diagnostics and cache hit-rate counters."""
    return _index_health_payload(_runtime())


@app.post("/api/index/rebuild", response_model=ReindexResponse, dependencies=[Depends(_require_roles("admin"))])
def post_index_rebuild(payload: ReindexRequest) -> ReindexResponse:
    """Triggers forced index rebuild and returns rebuilt index health."""
    runtime = _rebuild_runtime(refresh_prompts=payload.refresh_prompts)
    return ReindexResponse(ok=True, index_health=_index_health_payload(runtime))


@app.get("/api/metrics/sla", dependencies=[Depends(_require_roles("admin"))])
def get_metrics_sla() -> dict[str, float | int]:
    """Returns SLA metrics (latency percentiles, rates, counts)."""
    return _runtime().observability_store.sla_metrics()


@app.get("/api/metrics/monthly-snapshots", dependencies=[Depends(_require_roles("admin"))])
def get_metrics_monthly_snapshots() -> dict[str, object]:
    """Creates a monthly snapshot and returns snapshots + session rollups."""
    runtime = _runtime()
    runtime.observability_store.create_monthly_snapshot()
    return {
        "snapshots": runtime.observability_store.monthly_snapshots(),
        "session_rollups": runtime.observability_store.session_rollups(limit=100),
    }


@app.get("/api/traces/{trace_id}", dependencies=[Depends(_require_roles("admin"))])
def get_trace(trace_id: str) -> dict[str, object] | None:
    """Retrieves a trace plus latest QA pair for that trace."""
    return _runtime().observability_store.get_trace(trace_id)


@app.get("/api/governance-items", dependencies=[Depends(_require_roles("admin"))])
def list_governance_items(status: str | None = None, limit: int = 200) -> dict[str, object]:
    """Lists governance items with optional status filter and count summary."""
    runtime = _runtime()
    return {
        "items": runtime.observability_store.list_governance_items(status=status, limit=limit),
        "counts": runtime.observability_store.governance_counts(),
    }


@app.post("/api/governance-items/{item_id}/status", dependencies=[Depends(_require_roles("admin"))])
def update_governance_item_status(item_id: int, status: str) -> dict[str, object]:
    """Updates lifecycle status of a governance item."""
    runtime = _runtime()
    runtime.agent_router_service.governance_service.set_status(item_id, status)
    return {"ok": True, "item_id": item_id, "status": status}


@app.get("/api/question-library", dependencies=[Depends(_require_roles("viewer", "analyst", "admin"))])
def get_question_library() -> dict[str, object]:
    """Returns top questions, generated questions, and approved library items."""
    runtime = _runtime()
    return {
        "top_questions": runtime.top_questions,
        "generated_questions": runtime.generated_questions,
        "approved_library": runtime.observability_store.list_governance_items(status="Approved", limit=200),
    }


@app.post("/api/feedback", response_model=FeedbackResponse, dependencies=[Depends(_require_roles("analyst", "admin"))])
def post_feedback(payload: FeedbackRequest, raw_request: Request) -> FeedbackResponse:
    """Persists per-answer user feedback for quality tuning."""
    runtime = _runtime()
    role = str(getattr(raw_request.state, "user_role", "viewer"))
    feedback_id = runtime.observability_store.record_feedback(
        trace_id=payload.trace_id,
        session_id=payload.session_id,
        user_role=role,
        score=payload.score,
        note=payload.note,
    )
    return FeedbackResponse(feedback_id=feedback_id, stored=True)


@app.get(
    "/api/analytics/unanswered",
    response_model=UnansweredAnalyticsResponse,
    dependencies=[Depends(_require_roles("analyst", "admin"))],
)
def get_unanswered_analytics(limit: int = 25) -> UnansweredAnalyticsResponse:
    """Returns unanswered-query analytics summary and recent examples."""
    payload = _runtime().observability_store.unanswered_analytics(limit=limit)
    return UnansweredAnalyticsResponse(
        counts=payload.get("counts", {}),
        recent=payload.get("recent", []),
        top_patterns=payload.get("top_patterns", []),
    )


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
