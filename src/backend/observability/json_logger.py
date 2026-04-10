"""json_logger.py — Append-only JSONL event logger for the PMG chatbot.

Each call to :meth:`JsonEventLogger.log` appends one JSON-encoded line to
``data/logs/app_events.jsonl``.  The file is opened, written to, and closed
on every call so that log entries are durable even if the process crashes
mid-session.

The ``ts`` field is injected automatically as an ISO-8601 UTC timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class JsonEventLogger:
    """Append-only JSONL logger that writes one event per line.

    Parameters
    ----------
    logs_dir:
        Directory where ``app_events.jsonl`` is written.  Created if absent.

    Attributes
    ----------
    log_file:
        Resolved path to the JSONL file (set during ``__post_init__``).
    """

    logs_dir: Path

    def __post_init__(self) -> None:
        """Ensure the log directory exists and resolve the JSONL file path."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / "app_events.jsonl"

    def log(self, event: dict[str, object]) -> None:
        """Append a timestamped event as a single JSON line to the log file.

        The ``ts`` key (ISO-8601 UTC timestamp) is prepended automatically.
        All other keys from *event* follow in the order provided.

        Parameters
        ----------
        event:
            Dictionary of event fields.  Should at minimum include
            ``session_id``, ``trace_id``, ``event_type``, ``route_used``,
            ``status``, ``latency_ms``, and ``payload``.
        """
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **event,
        }
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
