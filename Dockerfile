# Dockerfile
# Multi-stage build for development and production
# Phase 2: Backend Framework & Basic API Services
# ===================================

# Use NVIDIA CUDA 12.1 base image with Ubuntu 22.04
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Metadata
LABEL maintainer="SD Multi-Modal Platform Team"
LABEL version="1.0.0-phase1"
LABEL description="Production-ready multi-model text-to-image platform"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive


# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python environment
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # Python build tools
    build-essential \
    cmake \
    # Image processing libraries
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # System utilities
    curl \
    wget \
    git \
    unzip \
    # Clean up unnecessary packages
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build Python environment
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install essential Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# ============================================================================
# Development stage
# ============================================================================
FROM base as development

# Set working directory
WORKDIR /app

# Create non-root user for development
RUN useradd --create-home --shell /bin/bash --uid 1000 devuser

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install development dependencies
RUN pip install \
    pytest \
    pytest-asyncio \
    httpx \
    black \
    isort \
    flake8 \
    mypy

# Create necessary directories
RUN mkdir -p /app/models /app/outputs /app/assets /app/logs && \
    chown -R devuser:devuser /app

# Copy application code
COPY --chown=devuser:devuser . .

# Switch to non-root user
USER devuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/simple || exit 1

# Default command for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# Production stage
# ============================================================================
FROM base as production

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install production-only dependencies
RUN pip install gunicorn

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/outputs /app/assets /app/logs && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Remove development files
RUN rm -rf tests/ scripts/ .git/ .env.example

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/simple || exit 1

# Production command with gunicorn
CMD ["gunicorn", "app.main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300"]

# ============================================================================
# Testing stage
# ============================================================================
FROM development as testing

# Install additional testing tools
RUN pip install \
    coverage \
    pytest-cov \
    pytest-benchmark

# Copy test configuration
COPY pytest.ini .

# Default command for testing
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=app", "--cov-report=html", "--cov-report=term-missing"]


# ============================================================================