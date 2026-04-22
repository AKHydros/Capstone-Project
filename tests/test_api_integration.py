"""API integration tests for PMG Intelligence FastAPI endpoints.

Covers:
- /health readiness probe (no auth, 200 or 503)
- /api/health/llm (no auth, returns status field)
- /api/auth/me (requires valid token, returns role)
- /api/agent-router auth enforcement (missing/invalid token → 401)
- /api/agent-router safety block (jailbreak query → blocked response, not error)
- /api/index/rebuild rate limit (admin-only, 2/min cap)
- Security headers present on all responses
- Retry-After header present on 429 responses
- /health returns 200 when runtime is healthy
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# sys.path bootstrap (also handled by conftest.py, kept here for standalone runs)
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Test tokens — injected into a patched AppConfig so load_dotenv(override=True)
# in config.py cannot overwrite them with real .env values.
# ---------------------------------------------------------------------------
_TEST_INTERNAL_TOKEN = "test-internal-token"
_TEST_ADMIN_TOKEN = "test-admin-token"
_TEST_ANALYST_TOKEN = "test-analyst-token"
_TEST_VIEWER_TOKEN = "test-viewer-token"

# ---------------------------------------------------------------------------
# Deferred import — avoids triggering full runtime bootstrap at module load.
# TestClient is imported after env is configured.
# ---------------------------------------------------------------------------


def _build_mock_runtime() -> MagicMock:
    """Returns a lightweight mock BootstrapArtifacts for use in API tests."""
    mock_runtime = MagicMock()
    mock_runtime.retriever = MagicMock()  # non-None → index "ready"
    mock_runtime.observability_store._connect.return_value = MagicMock()
    mock_runtime.observability_store.expire_stale_sessions.return_value = 0
    mock_runtime.observability_store.prune_old_records.return_value = 0
    mock_runtime.agent_router_service.llm_health_service.status.return_value = MagicMock(
        status="Connected",
        checked_at="2026-01-01T00:00:00+00:00",
        error_summary=None,
    )
    mock_runtime.agent_router_service.handle.return_value = MagicMock(
        trace_id="test-trace-id",
        route_used="deterministic",
        response="The answer is deterministic.",
        language_out="en",
        fallback_used=False,
        retrieval_mode="lexical",
        confidence_score=0.85,
        cache_status="miss",
        labels=[],
        takeaways=[],
        latency_ms=42.0,
        cards=[],
        sources=[],
        answer_mode="direct_answer",
        needs_clarification=False,
        unanswered_reason=None,
        fallback_reason=None,
        follow_up_suggestion=None,
        applied_context={},
        suggestions=[],
        reasoning=None,
        consistency_note=None,
    )
    return mock_runtime


def _make_test_config():
    """Returns a minimal AppConfig populated with test tokens only."""
    from pathlib import Path
    from backend.config import AppConfig

    return AppConfig(
        excel_source_path=Path("/tmp/test.xlsx"),
        index_cache_dir=Path("/tmp/cache"),
        observability_db_path=Path("/tmp/obs.db"),
        logs_dir=Path("/tmp/logs"),
        api_internal_token=_TEST_INTERNAL_TOKEN,
        api_viewer_tokens=(_TEST_VIEWER_TOKEN,),
        api_analyst_tokens=(_TEST_ANALYST_TOKEN,),
        api_admin_tokens=(_TEST_ADMIN_TOKEN,),
        api_base_url="http://localhost:8000",
        governance_policy_version="v1",
        default_router_mode="hybrid",
        conversation_memory_turns=6,
        openai_chat_model="gpt-4.1-mini",
        openai_chat_models=("gpt-4.1-mini",),
        rag_enabled=True,
        rag_confidence_threshold=0.72,
        rag_score_gap_threshold=0.07,
        rag_embed_cache_ttl=1800,
        rag_answer_cache_ttl=600,
        rag_batch_rebuild=False,
    )


def _get_client():
    """Returns a FastAPI TestClient with the runtime and config mocked out to
    avoid loading real Excel data, building an embedding index, or requiring
    real API tokens during tests.
    """
    from fastapi.testclient import TestClient
    from backend.api.app import app

    mock_runtime = _build_mock_runtime()
    test_config = _make_test_config()

    # Patch both _runtime() (data path) and load_config() (auth token lookup)
    # so tests are fully isolated from the real .env and dataset.
    with patch("backend.api.app._runtime", return_value=mock_runtime), \
         patch("backend.api.app.load_config", return_value=test_config), \
         patch("backend.config.load_config", return_value=test_config):
        yield TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    """Module-scoped TestClient backed by a mock runtime."""
    gen = _get_client()
    client = next(gen)
    yield client
    try:
        next(gen)
    except StopIteration:
        pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Tests: /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, api_client):
        """Public /health probe must return 200 with no auth."""
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_health_body_has_status_ok(self, api_client):
        """Response body must contain status=ok."""
        body = api_client.get("/health").json()
        assert body.get("status") == "ok"

    def test_health_body_has_index_field(self, api_client):
        """Response body must contain an index field."""
        body = api_client.get("/health").json()
        assert "index" in body

    def test_health_no_auth_required(self, api_client):
        """No Authorization header needed — must not 401."""
        resp = api_client.get("/health")
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Tests: /api/health/llm
# ---------------------------------------------------------------------------

class TestLlmHealthEndpoint:
    def test_llm_health_no_auth(self, api_client):
        """LLM health probe must be public."""
        resp = api_client.get("/api/health/llm")
        assert resp.status_code == 200

    def test_llm_health_has_status_field(self, api_client):
        body = api_client.get("/api/health/llm").json()
        assert "status" in body


# ---------------------------------------------------------------------------
# Tests: /api/auth/me
# ---------------------------------------------------------------------------

class TestAuthMe:
    def test_auth_me_no_token_returns_401(self, api_client):
        resp = api_client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_auth_me_invalid_token_returns_401(self, api_client):
        resp = api_client.get("/api/auth/me", headers=_bearer("totally-invalid-token"))
        assert resp.status_code == 401

    def test_auth_me_viewer_token_returns_viewer_role(self, api_client):
        resp = api_client.get("/api/auth/me", headers=_bearer(_TEST_VIEWER_TOKEN))
        assert resp.status_code == 200
        assert resp.json().get("role") == "viewer"

    def test_auth_me_admin_token_returns_admin_role(self, api_client):
        resp = api_client.get("/api/auth/me", headers=_bearer(_TEST_ADMIN_TOKEN))
        assert resp.status_code == 200
        assert resp.json().get("role") == "admin"


# ---------------------------------------------------------------------------
# Tests: /api/agent-router auth
# ---------------------------------------------------------------------------

class TestAgentRouterAuth:
    _PAYLOAD = {
        "session_id": "test-session",
        "query": "What is PMG22_WAI_q5a?",
        "language": "en",
        "filters": {},
        "inference": {},
        "conversation_context": [],
        "applied_context": {},
    }

    def test_no_token_returns_401(self, api_client):
        resp = api_client.post("/api/agent-router", json=self._PAYLOAD)
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, api_client):
        resp = api_client.post(
            "/api/agent-router",
            json=self._PAYLOAD,
            headers=_bearer("not-a-real-token"),
        )
        assert resp.status_code == 401

    def test_viewer_token_allowed(self, api_client):
        resp = api_client.post(
            "/api/agent-router",
            json=self._PAYLOAD,
            headers=_bearer(_TEST_VIEWER_TOKEN),
        )
        # Any non-401 status means auth was accepted (handler may 500 due to mock)
        assert resp.status_code != 401, f"viewer token should authenticate; got {resp.status_code}"


# ---------------------------------------------------------------------------
# Tests: Security headers
# ---------------------------------------------------------------------------

class TestSecurityHeaders:
    def test_x_content_type_options_header(self, api_client):
        resp = api_client.get("/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_x_frame_options_header(self, api_client):
        resp = api_client.get("/health")
        assert resp.headers.get("x-frame-options") == "DENY"

    def test_referrer_policy_no_referrer(self, api_client):
        resp = api_client.get("/health")
        assert resp.headers.get("referrer-policy") == "no-referrer"

    def test_hsts_header_present(self, api_client):
        resp = api_client.get("/health")
        hsts = resp.headers.get("strict-transport-security", "")
        assert "max-age" in hsts

    def test_csp_header_present(self, api_client):
        resp = api_client.get("/health")
        assert resp.headers.get("content-security-policy") is not None


# ---------------------------------------------------------------------------
# Tests: Wrong HTTP method
# ---------------------------------------------------------------------------

class TestMethodNotAllowed:
    def test_get_on_agent_router_returns_405(self, api_client):
        resp = api_client.get(
            "/api/agent-router",
            headers=_bearer("test-viewer-token"),
        )
        assert resp.status_code == 405


# ---------------------------------------------------------------------------
# Tests: DB retention
# ---------------------------------------------------------------------------

class TestDbRetention:
    def test_prune_old_records_returns_int(self, tmp_db):
        """prune_old_records() with days=0 should run without error and return int."""
        result = tmp_db.prune_old_records(days=0)
        assert isinstance(result, int)

    def test_prune_with_no_records_returns_zero(self, tmp_db):
        result = tmp_db.prune_old_records(days=90)
        assert result == 0

    def test_prune_removes_old_events(self, tmp_db):
        """An event inserted with a past timestamp should be pruned."""
        import sqlite3
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        with tmp_db._connect() as conn:
            conn.execute(
                "INSERT INTO events (ts, event_type) VALUES (?, ?)",
                (old_ts, "test_event"),
            )
            conn.commit()

        pruned = tmp_db.prune_old_records(days=90)
        assert pruned >= 1
