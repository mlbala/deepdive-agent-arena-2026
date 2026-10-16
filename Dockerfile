# syntax=docker/dockerfile:1.7
# DeepDive — multi-stage production image (linux/amd64 and linux/arm64).
#   docker build -t deepdive .
#   docker run --rm -p 8000:8000 --env-file .env deepdive

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- builder ---
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Toolchain only for wheels that must compile from source on less common architectures.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------- runtime ---
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="DeepDive" \
      org.opencontainers.image.description="Self-verifying multi-agent research & claim-entailment system (Agent Arena 2026, Track 01)" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    DATABASE_URL="sqlite:////app/data/deepdive.db" \
    ANONYMIZED_TELEMETRY="False"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin deepdive

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=deepdive:deepdive src ./src
COPY --chown=deepdive:deepdive eval ./eval
COPY --chown=deepdive:deepdive pyproject.toml README.md LICENSE ./
RUN mkdir -p /app/data && chown deepdive:deepdive /app/data

USER deepdive
EXPOSE 8000
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
