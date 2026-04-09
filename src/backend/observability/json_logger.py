from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class JsonEventLogger:
    logs_dir: Path

    def __post_init__(self) -> None:
        """Ensures log directory exists and sets JSONL file path."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / "app_events.jsonl"

    def log(self, event: dict[str, object]) -> None:
        """Appends timestamped event JSON line to log file."""
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **event,
        }
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
