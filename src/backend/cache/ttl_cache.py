from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    expired: int
    size: int


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1024) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: dict[str, tuple[float, T]] = {}
        self._hits = 0
        self._misses = 0
        self._expired = 0
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, object]:
        # Locks are not pickleable; serialize only data fields.
        return {
            "ttl_seconds": self.ttl_seconds,
            "max_size": self.max_size,
            "_store": self._store,
            "_hits": self._hits,
            "_misses": self._misses,
            "_expired": self._expired,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.ttl_seconds = int(state.get("ttl_seconds", 300))
        self.max_size = int(state.get("max_size", 1024))
        self._store = dict(state.get("_store", {}))
        self._hits = int(state.get("_hits", 0))
        self._misses = int(state.get("_misses", 0))
        self._expired = int(state.get("_expired", 0))
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self._misses += 1
                return None
            expires_at, value = item
            if expires_at < now:
                self._expired += 1
                self._misses += 1
                self._store.pop(key, None)
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: T) -> None:
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            if len(self._store) >= self.max_size:
                oldest_key = min(self._store.items(), key=lambda item: item[1][0])[0]
                self._store.pop(oldest_key, None)
            self._store[key] = (expires_at, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                expired=self._expired,
                size=len(self._store),
            )
