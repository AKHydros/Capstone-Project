from __future__ import annotations

import multiprocessing
import os


def _int_env(name: str, default: int, *, min_value: int, max_value: int | None = None) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


cpu_count = max(1, multiprocessing.cpu_count())
recommended_workers = (cpu_count * 2) + 1
max_workers = _int_env("API_MAX_WORKERS", recommended_workers, min_value=1)

bind = os.getenv("API_BIND", "0.0.0.0:8000")
worker_class = "uvicorn.workers.UvicornWorker"
workers = _int_env("API_WORKERS", recommended_workers, min_value=1, max_value=max_workers)
threads = _int_env("API_THREADS", 1, min_value=1, max_value=8)
backlog = _int_env("API_BACKLOG", 2048, min_value=128, max_value=65535)
keepalive = _int_env("API_KEEPALIVE", 10, min_value=2, max_value=120)
timeout = _int_env("API_TIMEOUT", 60, min_value=15, max_value=300)
graceful_timeout = _int_env("API_GRACEFUL_TIMEOUT", 30, min_value=10, max_value=120)
max_requests = _int_env("API_MAX_REQUESTS", 2000, min_value=100, max_value=100000)
max_requests_jitter = _int_env("API_MAX_REQUESTS_JITTER", 250, min_value=0, max_value=10000)

reuse_port = True
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("API_LOG_LEVEL", "info")
