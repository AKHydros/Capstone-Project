from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


class ObservabilityStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
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
                    completion_status TEXT DEFAULT 'open'
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
                """
            )
            conn.commit()

    def record_session_start(self, session_id: str, locale: str, consent: bool) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, started_at, locale, consent, completion_status)
                VALUES(?, ?, ?, ?, 'open')
                ON CONFLICT(session_id) DO UPDATE SET
                    locale=excluded.locale,
                    consent=excluded.consent
                """,
                (session_id, now, locale, int(consent)),
            )
            conn.commit()

    def session_has_consent(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT consent FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            return False
        return bool(row["consent"])

    def record_session_end(self, session_id: str, completion_status: str = "completed") -> None:
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
    ) -> None:
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
        with self._connect() as conn:
            conn.execute(
                "UPDATE governance_items SET status=?, updated_at=? WHERE id=?",
                (status, _now(), item_id),
            )
            conn.commit()

    def list_governance_items(self, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
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
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as c FROM governance_items GROUP BY status"
            ).fetchall()
        counts = {"Draft": 0, "Approved": 0, "Deprecated": 0}
        for row in rows:
            counts[str(row["status"])] = int(row["c"])
        return counts

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
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
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM monthly_snapshots ORDER BY month_key DESC").fetchall()
        return [dict(r) for r in rows]

    def recent_qa_pairs(self, limit: int = 100) -> list[dict[str, Any]]:
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
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 1:
        return values[-1]
    idx = int(round((len(values) - 1) * p))
    return values[idx]
