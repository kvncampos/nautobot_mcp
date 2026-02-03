# Nautobot MCP Server Dockerfile
# Multi-stage build for optimal image size and security

FROM python:3.11-slim AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash mcpuser

# Set working directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder --chown=mcpuser:mcpuser /app/.venv /app/.venv

# Copy application code
COPY --chown=mcpuser:mcpuser . .

# Create directories for ChromaDB data persistence
RUN mkdir -p /app/backend/chroma_db /app/backend/models \
    && chown -R mcpuser:mcpuser /app/backend

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # ChromaDB configuration
    TOKENIZERS_PARALLELISM=false \
    # MCP configuration (can be overridden at runtime)
    MCP_TRANSPORT=stdio \
    MCP_PORT=8000

# Switch to non-root user
USER mcpuser

# Expose HTTP port (only used in HTTP mode)
EXPOSE 8000

# Default entrypoint
ENTRYPOINT ["python", "main.py"]

# Default command (can be overridden)
CMD ["--mode", "stdio"]
