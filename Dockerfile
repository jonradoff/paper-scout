# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml .
# Install CPU-only PyTorch first (skips ~3.5GB of CUDA libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/
COPY interest_profile/ interest_profile/
COPY scripts/ scripts/

RUN mkdir -p /data/models

ENV PYTHONPATH=/app/src
ENV MODEL_CACHE_DIR=/data/models
ENV PYTHONUNBUFFERED=1

# Default: run the scheduler (daily 5 AM EST pipeline)
CMD ["python", "-m", "paper_scout.scheduler"]
