# Nautobot MCP Server Dockerfile
# Multi-stage build for optimal image size and security
# Optimized for uv package manager with cache mounts and intermediate layers

FROM python:3.11-slim AS builder

# Copy uv from official image (pinned version for reproducibility)
# Version 0.5.21 matches local development environment
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set uv environment variables optimized for Docker
# UV_COMPILE_BYTECODE: Precompile Python bytecode for faster startup
# UV_LINK_MODE: Use copy instead of hardlinks (required for cache mounts)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set working directory
WORKDIR /app

# Install dependencies first (with cache mount and bind mounts)
# This layer is cached separately from project code for better rebuild performance
# Bind mounts prevent copying files into the image, improving layer efficiency
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy project source code
COPY . /app

# Install the project itself (with cache mount)
# This completes the installation by installing the project into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

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
    # UV configuration for production
    UV_NO_DEV=1 \
    # ChromaDB configuration
    TOKENIZERS_PARALLELISM=false \
    # MCP configuration (can be overridden at runtime)
    MCP_TRANSPORT=stdio \
    MCP_PORT=8081

# Switch to non-root user
USER mcpuser

# Expose HTTP port (only used in HTTP mode)
EXPOSE 8081

# Default entrypoint
ENTRYPOINT ["python", "main.py"]

# Default command (can be overridden)
CMD ["--mode", "stdio"]
