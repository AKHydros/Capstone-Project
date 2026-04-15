from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any


class ObservabilityStore:
    def __init__(self, db_path: Path) -> None:
        """Initializes DB path and ensures observability schema exists."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Returns thread-local SQLite connection with performance PRAGMAs."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-20000")
        conn.execute("PRAGMA mmap_size=134217728")
        conn.execute("PRAGMA busy_timeout=5000")
        self._local.conn = conn
        return conn

    def close(self) -> None:
        """Closes and clears current thread-local DB connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _init_schema(self) -> None:
        """Creates required tables and indexes for sessions/events/traces/governance."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    locale TEXT,
                    consent INTEGER DEFAULT 0,
                    completion_status TEXT DEFAULT 'open',
                    last_active_at TEXT,
                    token_hash TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    session_id TEXT,
                    trace_id TEXT,
                    event_type TEXT NOT NULL,
                    route_used TEXT,
                    status TEXT,
                    latency_ms REAL,
                    payload_json TEXT,
                    error_text TEXT
                );

                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    route_used TEXT,
                    fallback_used INTEGER,
                    latency_ms REAL,
                    status TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    session_id TEXT,
                    query_text TEXT,
                    response_text TEXT,
                    language_in TEXT,
                    language_out TEXT,
                    labels_json TEXT,
                    takeaways_json TEXT,
                    route_used TEXT,
                    fallback_used INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS consent_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_consent INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    locale TEXT,
                    effective_logging_level TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS governance_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_text TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    labels_json TEXT,
                    takeaways_json TEXT,
                    last_trace_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS monthly_snapshots (
                    month_key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    p50_latency REAL,
                    p95_latency REAL,
                    error_rate REAL,
                    fallback_rate REAL,
                    session_completion_rate REAL,
                    total_sessions INTEGER,
                    total_queries INTEGER
                );

                CREATE TABLE IF NOT EXISTS response_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS unanswered_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS saved_searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_norm TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    applied_context_json TEXT NOT NULL,
                    similarity_key TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    reopen_count INTEGER NOT NULL DEFAULT 0,
                    last_reopened_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_trace_id ON events(trace_id);
                CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id);
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
                CREATE INDEX IF NOT EXISTS idx_traces_session_id ON traces(session_id);
                CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at);
                CREATE INDEX IF NOT EXISTS idx_traces_fallback ON traces(fallback_used);
                CREATE INDEX IF NOT EXISTS idx_qa_pairs_trace_id ON qa_pairs(trace_id);
                CREATE INDEX IF NOT EXISTS idx_qa_pairs_session_id ON qa_pairs(session_id);
                CREATE INDEX IF NOT EXISTS idx_qa_pairs_created_at ON qa_pairs(created_at);
                CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
                CREATE INDEX IF NOT EXISTS idx_sessions_completion_status ON sessions(completion_status);
                CREATE INDEX IF NOT EXISTS idx_governance_status_updated ON governance_items(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_trace_id ON response_feedback(trace_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON response_feedback(created_at);
                CREATE INDEX IF NOT EXISTS idx_unanswered_reason ON unanswered_queries(reason);
                CREATE INDEX IF NOT EXISTS idx_unanswered_created_at ON unanswered_queries(created_at);
                CREATE INDEX IF NOT EXISTS idx_saved_searches_created_at ON saved_searches(created_at);
                CREATE INDEX IF NOT EXISTS idx_saved_searches_session_id ON saved_searches(session_id);
                CREATE INDEX IF NOT EXISTS idx_saved_searches_similarity ON saved_searches(similarity_key);
                CREATE INDEX IF NOT EXISTS idx_saved_searches_pinned ON saved_searches(pinned);
                """
            )
            conn.commit()
            # --- migrations for existing databases ---
            # Run ALTER TABLE first so that any index referencing new columns
            # (e.g. idx_sessions_last_active) is created AFTER the column exists.
            for _col, _ddl in (
                ("last_active_at", "ALTER TABLE sessions ADD COLUMN last_active_at TEXT"),
                ("token_hash",     "ALTER TABLE sessions ADD COLUMN token_hash TEXT"),
            ):
                try:
                    conn.execute(_ddl)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # column already exists
            # Create indexes that depend on migrated columns (safe to re-run).
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active_at)"
                )
                conn.commit()
            except sqlite3.OperationalError:
                pass

    def record_session_start(
        self, session_id: str, locale: str, consent: bool, token_hash: str = ""
    ) -> None:
        """Inserts session if absent with locale/consent state and owner token hash."""
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, started_at, locale, consent, completion_status,
                                     last_active_at, token_hash)
                VALUES(?, ?, ?, ?, 'open', ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET last_active_at=excluded.last_active_at
                """,
                (session_id, now, locale, int(consent), now, token_hash or None),
            )
            conn.commit()

    def ensure_session_and_get_consent(self, session_id: str, locale: str) -> bool:
        """Ensures session row exists, bumps last_active_at, and returns current consent flag."""
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT consent FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE sessions SET last_active_at=? WHERE session_id=?",
                    (now, session_id),
                )
                conn.commit()
                return bool(row["consent"])

            conn.execute(
                """
                INSERT INTO sessions(session_id, started_at, locale, consent, completion_status,
                                     last_active_at)
                VALUES(?, ?, ?, ?, 'open', ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, now, locale, 0, now),
            )
            conn.commit()
            return False

    def session_has_consent(self, session_id: str) -> bool:
        """Reads consent flag for session."""
        with self._connect() as conn:
            row = conn.execute("SELECT consent FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            return False
        return bool(row["consent"])

    def session_owner_hash(self, session_id: str) -> str | None:
        """Returns the token_hash for a session, or None if unset / session absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT token_hash FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return str(row["token_hash"]) if row["token_hash"] else None

    def expire_stale_sessions(self, *, idle_hours: int = 24) -> int:
        """Marks sessions that have been idle for more than *idle_hours* as 'expired'.

        A session is considered idle when ``last_active_at`` (or ``started_at``
        as a fallback) is older than the given threshold.  Returns the number
        of sessions updated.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=idle_hours)).isoformat(
            timespec="seconds"
        )
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE sessions
                SET completion_status='expired', ended_at=?
                WHERE completion_status='open'
                  AND COALESCE(last_active_at, started_at) < ?
                """,
                (_now(), cutoff),
            )
            conn.commit()
            return cur.rowcount

    def record_session_end(self, session_id: str, completion_status: str = "completed") -> None:
        """Marks session end time and completion status."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at=?, completion_status=? WHERE session_id=?",
                (_now(), completion_status, session_id),
            )
            conn.commit()

    def record_event(
        self,
        *,
        session_id: str,
        trace_id: str | None,
        event_type: str,
        route_used: str | None,
        status: str,
        latency_ms: float | None,
        payload: dict[str, Any] | None,
        error_text: str | None,
    ) -> None:
        """Inserts event record with payload/error details."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events(ts, session_id, trace_id, event_type, route_used, status, latency_ms, payload_json, error_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    session_id,
                    trace_id,
                    event_type,
                    route_used,
                    status,
                    latency_ms,
                    json.dumps(payload or {}),
                    error_text,
                ),
            )
            conn.commit()

    def record_trace(
        self,
        *,
        trace_id: str,
        session_id: str,
        route_used: str,
        fallback_used: bool,
        latency_ms: float,
        status: str,
    ) -> None:
        """Upserts trace record and latency/fallback metadata."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traces(trace_id, session_id, route_used, fallback_used, latency_ms, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    route_used=excluded.route_used,
                    fallback_used=excluded.fallback_used,
                    latency_ms=excluded.latency_ms,
                    status=excluded.status
                """,
                (trace_id, session_id, route_used, int(fallback_used), latency_ms, status, _now()),
            )
            conn.commit()

    def record_qa_pair(
        self,
        *,
        trace_id: str,
        session_id: str,
        query_text: str,
        response_text: str,
        language_in: str,
        language_out: str,
        labels: list[str],
        takeaways: list[str],
        route_used: str,
        fallback_used: bool,
        logging_level: str = "full",
    ) -> None:
        """Inserts query/response pair for audit and analytics.

        PII is always scrubbed from stored text using the same regex patterns
        as ``SafetyService.redact_pii``.  When ``logging_level`` is ``"minimal"``
        the response body is replaced with a placeholder — only the query
        (post-redaction) and metadata are retained for analytics purposes.
        """
        query_text = _redact_pii(query_text)
        if logging_level == "minimal":
            response_text = "[redacted — minimal logging consent]"
        else:
            response_text = _redact_pii(response_text)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO qa_pairs(trace_id, session_id, query_text, response_text, language_in, language_out, labels_json, takeaways_json, route_used, fallback_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    session_id,
                    query_text,
                    response_text,
                    language_in,
                    language_out,
                    json.dumps(labels),
                    json.dumps(takeaways),
                    route_used,
                    int(fallback_used),
                    _now(),
                ),
            )
            conn.commit()

    def record_consent(
        self,
        *,
        session_id: str,
        user_consent: bool,
        timestamp: str,
        locale: str,
        effective_logging_level: str,
    ) -> int:
        """Inserts consent record and updates session consent state."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO consent_records(session_id, user_consent, timestamp, locale, effective_logging_level)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, int(user_consent), timestamp, locale, effective_logging_level),
            )
            conn.execute(
                "UPDATE sessions SET consent=? WHERE session_id=?",
                (int(user_consent), session_id),
            )
            conn.commit()
            return int(cur.lastrowid)

    def record_feedback(
        self,
        *,
        trace_id: str,
        session_id: str,
        user_role: str,
        score: int,
        note: str | None,
    ) -> int:
        """Persists per-response feedback (thumbs + optional note)."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO response_feedback(trace_id, session_id, user_role, score, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (trace_id, session_id, user_role, int(score), (note or "").strip() or None, _now()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def recent_feedback(self, limit: int = 100) -> list[dict[str, Any]]:
        """Returns recent feedback records for operator review."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM response_feedback ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_unanswered(
        self,
        *,
        trace_id: str,
        session_id: str,
        user_role: str,
        reason: str,
        query_text: str,
    ) -> int:
        """Stores unanswered-query classification for analytics."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO unanswered_queries(trace_id, session_id, user_role, reason, query_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (trace_id, session_id, user_role, reason, query_text, _now()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def unanswered_analytics(self, limit: int = 25) -> dict[str, Any]:
        """Returns aggregate unanswered counts, recent examples, and top repeated patterns."""
        with self._connect() as conn:
            reason_rows = conn.execute(
                "SELECT reason, COUNT(*) as c FROM unanswered_queries GROUP BY reason"
            ).fetchall()
            recent_rows = conn.execute(
                """
                SELECT trace_id, session_id, user_role, reason, query_text, created_at
                FROM unanswered_queries
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            top_pattern_rows = conn.execute(
                """
                SELECT LOWER(TRIM(query_text)) as query_pattern, COUNT(*) as c
                FROM unanswered_queries
                GROUP BY LOWER(TRIM(query_text))
                ORDER BY c DESC, query_pattern ASC
                LIMIT 10
                """
            ).fetchall()

        counts = {"no_cards": 0, "clarifier_only": 0}
        for row in reason_rows:
            reason = str(row["reason"])
            counts[reason] = int(row["c"])

        recent = [
            {
                "trace_id": str(row["trace_id"]),
                "session_id": str(row["session_id"]),
                "user_role": str(row["user_role"]),
                "reason": str(row["reason"]),
                "query_text": str(row["query_text"]),
                "created_at": str(row["created_at"]),
            }
            for row in recent_rows
        ]
        top_patterns = [
            {
                "query_pattern": str(row["query_pattern"] or ""),
                "count": int(row["c"]),
            }
            for row in top_pattern_rows
            if str(row["query_pattern"] or "").strip()
        ]
        return {"counts": counts, "recent": recent, "top_patterns": top_patterns}

    def save_search(
        self,
        *,
        session_id: str,
        user_role: str,
        query_text: str,
        filters: dict[str, str | None] | None,
        applied_context: dict[str, str | None] | None,
        pinned: bool = False,
    ) -> int:
        """Persists a normalized search entry for reopen/reuse workflows."""
        now = _now()
        normalized_query = _normalize_query_text(query_text)
        normalized_filters = {
            key: (str(value).strip() if value is not None and str(value).strip() else None)
            for key, value in (filters or {}).items()
        }
        normalized_context = {
            key: (str(value).strip() if value is not None and str(value).strip() else None)
            for key, value in (applied_context or {}).items()
        }
        similarity_key = _similarity_key(normalized_query)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO saved_searches(
                    session_id, user_role, query_text, query_norm,
                    filters_json, applied_context_json, similarity_key,
                    pinned, reopen_count, last_reopened_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """,
                (
                    session_id,
                    user_role,
                    query_text.strip(),
                    normalized_query,
                    json.dumps(normalized_filters),
                    json.dumps(normalized_context),
                    similarity_key,
                    int(pinned),
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_saved_searches(
        self,
        *,
        limit: int = 25,
        session_id: str | None = None,
        user_role: str | None = None,
        pinned_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Returns recent saved searches with optional session/role filtering."""
        where: list[str] = []
        params: list[Any] = []
        if session_id:
            where.append("session_id=?")
            params.append(session_id)
        if user_role:
            where.append("user_role=?")
            params.append(user_role)
        if pinned_only:
            where.append("pinned=1")
        query = "SELECT * FROM saved_searches"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY pinned DESC, updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._saved_search_row_to_payload(row) for row in rows]

    def reopen_saved_search(self, *, search_id: int) -> dict[str, Any] | None:
        """Marks a saved search as reopened and returns the updated payload."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM saved_searches WHERE id=?",
                (search_id,),
            ).fetchone()
            if row is None:
                return None
            now = _now()
            conn.execute(
                """
                UPDATE saved_searches
                SET reopen_count=reopen_count+1, last_reopened_at=?, updated_at=?
                WHERE id=?
                """,
                (now, now, search_id),
            )
            updated = conn.execute("SELECT * FROM saved_searches WHERE id=?", (search_id,)).fetchone()
            conn.commit()
        if updated is None:
            return None
        return self._saved_search_row_to_payload(updated)

    def _saved_search_row_to_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        """Converts raw saved-search row to API-safe payload."""
        return {
            "id": int(row["id"]),
            "session_id": str(row["session_id"]),
            "user_role": str(row["user_role"]),
            "query_text": str(row["query_text"]),
            "query_norm": str(row["query_norm"]),
            "filters": json.loads(row["filters_json"] or "{}"),
            "applied_context": json.loads(row["applied_context_json"] or "{}"),
            "similarity_key": str(row["similarity_key"]),
            "pinned": bool(row["pinned"]),
            "reopen_count": int(row["reopen_count"]),
            "last_reopened_at": str(row["last_reopened_at"]) if row["last_reopened_at"] else None,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def upsert_governance_item(
        self,
        *,
        question_text: str,
        source: str,
        status: str,
        labels: list[str],
        takeaways: list[str],
        last_trace_id: str | None,
    ) -> int:
        """Upserts governance item by question text and returns item ID."""
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO governance_items(question_text, source, status, labels_json, takeaways_json, last_trace_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(question_text) DO UPDATE SET
                    source=excluded.source,
                    status=excluded.status,
                    labels_json=excluded.labels_json,
                    takeaways_json=excluded.takeaways_json,
                    last_trace_id=excluded.last_trace_id,
                    updated_at=excluded.updated_at
                """,
                (
                    question_text,
                    source,
                    status,
                    json.dumps(labels),
                    json.dumps(takeaways),
                    last_trace_id,
                    now,
                    now,
                ),
            )
            cur = conn.execute(
                "SELECT id FROM governance_items WHERE question_text=?",
                (question_text,),
            )
            row = cur.fetchone()
            conn.commit()
            return int(row["id"]) if row else 0

    def update_governance_status(self, item_id: int, status: str) -> None:
        """Updates governance status and timestamp."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE governance_items SET status=?, updated_at=? WHERE id=?",
                (status, _now(), item_id),
            )
            conn.commit()

    def list_governance_items(self, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Lists governance items with deserialized labels/takeaways."""
        query = "SELECT * FROM governance_items"
        params: list[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": int(r["id"]),
                "question_text": r["question_text"],
                "source": r["source"],
                "status": r["status"],
                "labels": json.loads(r["labels_json"] or "[]"),
                "takeaways": json.loads(r["takeaways_json"] or "[]"),
                "last_trace_id": r["last_trace_id"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def governance_counts(self) -> dict[str, int]:
        """Returns aggregate governance counts by status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as c FROM governance_items GROUP BY status"
            ).fetchall()
        counts = {"Draft": 0, "Approved": 0, "Deprecated": 0}
        for row in rows:
            counts[str(row["status"])] = int(row["c"])
        return counts

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Returns trace record and latest QA payload for that trace."""
        with self._connect() as conn:
            tr = conn.execute("SELECT * FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
            if tr is None:
                return None
            qa = conn.execute(
                "SELECT * FROM qa_pairs WHERE trace_id=? ORDER BY id DESC LIMIT 1",
                (trace_id,),
            ).fetchone()

        payload = {
            "trace_id": tr["trace_id"],
            "session_id": tr["session_id"],
            "route_used": tr["route_used"],
            "fallback_used": bool(tr["fallback_used"]),
            "latency_ms": tr["latency_ms"],
            "status": tr["status"],
            "created_at": tr["created_at"],
        }
        if qa is not None:
            payload["qa"] = {
                "query_text": qa["query_text"],
                "response_text": qa["response_text"],
                "labels": json.loads(qa["labels_json"] or "[]"),
                "takeaways": json.loads(qa["takeaways_json"] or "[]"),
            }
        return payload

    def sla_metrics(self) -> dict[str, Any]:
        """Computes p50/p95 latency and operational rates/counters."""
        with self._connect() as conn:
            lat_rows = conn.execute(
                "SELECT latency_ms FROM events WHERE event_type='agent_router' AND latency_ms IS NOT NULL"
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) as c FROM events WHERE event_type='agent_router'").fetchone()["c"]
            errors = conn.execute(
                "SELECT COUNT(*) as c FROM events WHERE event_type='agent_router' AND status='error'"
            ).fetchone()["c"]
            fallbacks = conn.execute(
                "SELECT COUNT(*) as c FROM traces WHERE fallback_used=1"
            ).fetchone()["c"]
            sessions_total = conn.execute("SELECT COUNT(*) as c FROM sessions").fetchone()["c"]
            sessions_completed = conn.execute(
                "SELECT COUNT(*) as c FROM sessions WHERE completion_status='completed'"
            ).fetchone()["c"]

        latencies = sorted([float(r["latency_ms"]) for r in lat_rows])
        p50 = _percentile(latencies, 0.50)
        p95 = _percentile(latencies, 0.95)
        error_rate = (errors / total) if total else 0.0
        fallback_rate = (fallbacks / total) if total else 0.0
        completion_rate = (sessions_completed / sessions_total) if sessions_total else 0.0

        return {
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "error_rate": round(error_rate, 4),
            "fallback_rate": round(fallback_rate, 4),
            "session_completion_rate": round(completion_rate, 4),
            "total_sessions": int(sessions_total),
            "total_queries": int(total),
        }

    def create_monthly_snapshot(self, month_key: str | None = None) -> dict[str, Any]:
        """Saves monthly KPI snapshot from current SLA metrics."""
        month_key = month_key or datetime.now(timezone.utc).strftime("%Y-%m")
        metrics = self.sla_metrics()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO monthly_snapshots(month_key, created_at, p50_latency, p95_latency, error_rate, fallback_rate, session_completion_rate, total_sessions, total_queries)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(month_key) DO UPDATE SET
                    created_at=excluded.created_at,
                    p50_latency=excluded.p50_latency,
                    p95_latency=excluded.p95_latency,
                    error_rate=excluded.error_rate,
                    fallback_rate=excluded.fallback_rate,
                    session_completion_rate=excluded.session_completion_rate,
                    total_sessions=excluded.total_sessions,
                    total_queries=excluded.total_queries
                """,
                (
                    month_key,
                    _now(),
                    metrics["p50_latency_ms"],
                    metrics["p95_latency_ms"],
                    metrics["error_rate"],
                    metrics["fallback_rate"],
                    metrics["session_completion_rate"],
                    metrics["total_sessions"],
                    metrics["total_queries"],
                ),
            )
            conn.commit()
        return {"month_key": month_key, **metrics}

    def monthly_snapshots(self) -> list[dict[str, Any]]:
        """Returns monthly snapshot history."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM monthly_snapshots ORDER BY month_key DESC").fetchall()
        return [dict(r) for r in rows]

    def recent_qa_pairs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Returns recent QA records with decoded JSON fields."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_pairs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        payload: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["labels"] = json.loads(item.pop("labels_json", "[]") or "[]")
            item["takeaways"] = json.loads(item.pop("takeaways_json", "[]") or "[]")
            payload.append(item)
        return payload

    def session_rollups(self, limit: int = 100) -> list[dict[str, Any]]:
        """Returns session-level rollups joined with event counts."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.started_at, s.ended_at, s.locale, s.consent, s.completion_status,
                       COUNT(e.id) as event_count
                FROM sessions s
                LEFT JOIN events e ON s.session_id = e.session_id
                GROUP BY s.session_id
                ORDER BY s.started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def _now() -> str:
    """Returns current UTC ISO timestamp (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _percentile(values: list[float], p: float) -> float:
    """Returns percentile value from sorted latency list."""
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 1:
        return values[-1]
    idx = int(round((len(values) - 1) * p))
    return values[idx]


def _normalize_query_text(query_text: str) -> str:
    """Normalizes query text for lightweight similarity grouping."""
    return " ".join((query_text or "").strip().lower().split())


def _similarity_key(query_norm: str, *, max_tokens: int = 5) -> str:
    """Builds a short key from first tokens to group similar queries."""
    tokens = [token for token in query_norm.split(" ") if token]
    if not tokens:
        return "empty"
    return " ".join(tokens[:max_tokens])


def _redact_pii(text: str) -> str:
    """Redacts email addresses and phone numbers before persistence.

    Mirrors ``SafetyService.redact_pii`` without importing that class here
    to avoid a circular dependency between the observability and services layers.
    """
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    text = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[redacted-phone]", text)
    return text
