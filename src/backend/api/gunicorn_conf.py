from __future__ import annotations

import multiprocessing
import os


def _int_env(name: str, default: int, *, min_value: int, max_value: int | None = None) -> int:
    """Safely parses and clamps integer env vars for gunicorn settings."""
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

# ── Structured JSON logging ───────────────────────────────────────────────────
# When python-json-logger is installed, all gunicorn log output is emitted as
# newline-delimited JSON — machine-parseable by ELK, CloudWatch, Datadog, etc.
# Falls back to plaintext if the package is not available (e.g. dev installs).
def _json_log_config() -> dict | None:
    try:
        from pythonjsonlogger import jsonlogger  # noqa: F401
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": loglevel.upper(), "handlers": ["console"]},
            "loggers": {
                "gunicorn.error":  {"handlers": ["console"], "propagate": False},
                "gunicorn.access": {"handlers": ["console"], "propagate": False},
                "uvicorn":         {"handlers": ["console"], "propagate": False},
                "uvicorn.access":  {"handlers": ["console"], "propagate": False},
            },
        }
    except ImportError:
        return None


_log_cfg = _json_log_config()
if _log_cfg is not None:
    logconfig_dict = _log_cfg
