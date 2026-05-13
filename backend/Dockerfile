# Multi-stage build for Insurance Fraud Detection Backend
FROM python:3.13-slim AS builder

# Install uv for faster, reliable dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies to a virtual environment
# We use --no-dev to exclude development dependencies
RUN uv venv /opt/venv && \
    uv sync --no-dev --no-install-project

# Copy the rest of the application to install the project itself
COPY . .
RUN . /opt/venv/bin/activate && uv pip install -e .


# Stage 2: Runtime
FROM python:3.13-slim

# Standard Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install runtime dependencies: 
# - curl: for healthchecks (used by Docker Compose)
# - postgresql-client: for database debugging/readiness checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY models/ ./models/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check using httpx (which is a project dependency)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; import sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code == 200 else 1)"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
