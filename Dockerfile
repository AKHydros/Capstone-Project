# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Builds the Python wheel so the runtime stage has no build tools.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build toolchain
RUN pip install --no-cache-dir build==1.2.2

# Copy only the files needed to build the package
COPY pyproject.toml README.md ./
COPY src/ src/

RUN python -m build --wheel --outdir /build/dist

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="PMG Intelligence API"
LABEL org.opencontainers.image.description="FastAPI backend for conversational research data dictionary"
LABEL org.opencontainers.image.source="https://github.com/AKHydros/Capstone-Project"

# Create non-root user for least-privilege execution
RUN addgroup --system pmg && adduser --system --ingroup pmg pmg

WORKDIR /app

# Install runtime dependencies from the built wheel
COPY --from=builder /build/dist/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl \
        "gunicorn~=25.3.0" \
        "uvicorn[standard]~=0.42.0" \
        "python-json-logger~=2.0" \
    && rm -rf /tmp/wheels

# Copy gunicorn config (excluded from wheel by setuptools)
COPY src/backend/api/gunicorn_conf.py src/backend/api/gunicorn_conf.py

# Data directory — mounted as a volume in production
# Index cache, SQLite observability DB, and JSONL event logs live here.
RUN mkdir -p /app/data/cache /app/data/logs \
    && chown -R pmg:pmg /app/data

USER pmg

EXPOSE 8000

# Configurable via environment:
#   API_BIND            — listen address (default: 0.0.0.0:8000)
#   API_WORKERS         — gunicorn worker count (default: CPU*2+1)
#   API_TIMEOUT         — request timeout seconds (default: 60)
#   REQUEST_TIMEOUT_SECONDS — middleware hard ceiling (default: 55)
ENV API_BIND=0.0.0.0:8000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["gunicorn", "-c", "src/backend/api/gunicorn_conf.py", "src.backend.api.app:app"]
