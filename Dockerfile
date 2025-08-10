# Dockerfile
# SD Multi-Modal Platform Dockerfile
# Phase 1: Backend Infrastructure
# ===================================

# Use NVIDIA CUDA 12.1 base image with Ubuntu 22.04
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Metadata
LABEL maintainer="SD Multi-Modal Platform Team"
LABEL version="1.0.0-phase1"
LABEL description="Production-ready multi-model text-to-image platform"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Working directory
WORKDIR /app

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

# 複製需求檔案
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip3 install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY app/ ./app/
COPY services/ ./services/
COPY utils/ ./utils/
COPY scripts/ ./scripts/

# 複製配置檔案
COPY .env.example .env

# 建立必要目錄
RUN mkdir -p \
    models/stable-diffusion/sdxl \
    models/controlnet \
    models/lora \
    models/vae \
    models/cache \
    outputs/txt2img \
    outputs/metadata \
    assets/presets \
    logs

# 設定檔案權限
RUN chmod +x scripts/*.py

# 建立非 root 使用者 (安全考量)
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 環境變數
ENV PYTHONPATH=/app
ENV DEVICE=cuda
ENV LOG_LEVEL=INFO

# 啟動命令
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ===================================
# 建置指令範例:
# docker build -t sd-platform:phase1 .
#
# 執行指令範例:
# docker run --gpus all -p 8000:8000 \
#   -v $(pwd)/models:/app/models \
#   -v $(pwd)/outputs:/app/outputs \
#   -e DEVICE=cuda \
#   sd-platform:phase1
# ===================================