# app/api/v1/health.py
"""
SD Multi-Modal Platform - Health Check API
Phase 1: System Health Check and Diagnostics
"""
import os

import logging
import time
import platform
import psutil
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, Depends, Query
import torch
import pynvml as nvml

from app.config import Settings, settings, get_settings
from app.schemas.requests import HealthCheckRequest
from app.schemas.responses import HealthCheckResponse


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    check_gpu: bool = Query(default=True, description="Check GPU status"),
    check_models: bool = Query(default=False, description="Check model availability"),
    verbose: bool = Query(default=False, description="Include detailed system info"),
    config: Settings = Depends(get_settings),
) -> HealthCheckResponse:
    """
    Perform a comprehensive health check of the system.

    This endpoint checks:
    - Service status
    - System resources (CPU, memory, disk)
    - GPU status (if available)
    - File system paths
    - Model availability (if configured)
    """

    start_time = time.time()
    response = HealthCheckResponse(status="healthy")

    # === Service Info ===
    response.service_info = {
        "name": "SD Multi-Modal Platform",
        "version": "1.0.0-phase1",
        "phase": "Phase 1: MVP Backend",
        "uptime": time.time() - start_time,  # Uptime in seconds
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "primary_model": config.PRIMARY_MODEL,
        "device": config.DEVICE,
    }

    # === Service Status Check ===
    try:
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_used_percent = memory.percent

        # Disk info
        disk = psutil.disk_usage("/")
        disk_free_gb = disk.free / (1024**3)
        disk_used_percent = (disk.used / disk.total) * 100

        response.system_status = {
            "cpu": {
                "usage_percent": cpu_percent,
                "core_count": cpu_count,
                "status": "healthy" if cpu_percent < 80 else "warning",
            },
            "memory": {
                "total_gb": round(memory_gb, 2),
                "used_percent": memory_used_percent,
                "available_gb": round(memory.available / (1024**3), 2),
                "status": "healthy" if memory_used_percent < 85 else "warning",
            },
            "disk": {
                "free_gb": round(disk_free_gb, 2),
                "used_percent": round(disk_used_percent, 2),
                "status": "healthy" if disk_free_gb > 5 else "warning",
            },
        }

        response.checks["system_resources"] = True

        # CPU and memory warnings
        if cpu_percent > 80:
            response.add_warning(f"High CPU usage: {cpu_percent:.1f}%")
        if memory_used_percent > 85:
            response.add_warning(f"High memory usage: {memory_used_percent:.1f}%")
        if disk_free_gb < 5:
            response.add_warning(f"Low disk space: {disk_free_gb:.1f}GB remaining")

    except Exception as exc:
        logger.error(f"System resource check failed: {exc}")
        response.checks["system_resources"] = False
        response.add_warning("Failed to check system resources")

    # === GPU Check (if enabled) ===
    if check_gpu:
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)

                # GPU memory stats
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                gpu_memory_free = gpu_memory_total - gpu_memory_cached

                # GPU temperature (if available)
                try:
                    nvml.nvmlInit()
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_temp = nvml.nvmlDeviceGetTemperature(
                        handle, nvml.NVML_TEMPERATURE_GPU
                    )
                    nvml.nvmlShutdown()
                except:
                    gpu_temp = None

                response.gpu_status = {
                    "available": True,
                    "device_count": gpu_count,
                    "device_name": gpu_name,
                    "memory": {
                        "total_gb": round(gpu_memory_total, 2),
                        "allocated_gb": round(gpu_memory_allocated, 2),
                        "cached_gb": round(gpu_memory_cached, 2),
                        "free_gb": round(gpu_memory_free, 2),
                        "usage_percent": round(
                            (gpu_memory_cached / gpu_memory_total) * 100, 2
                        ),
                    },
                    "temperature": gpu_temp,
                    "status": "healthy" if gpu_memory_free > 2 else "warning",
                }

                response.checks["gpu_available"] = True

                # GPU warnings
                if gpu_memory_free < 2:
                    response.add_warning(
                        f"Low GPU memory: {gpu_memory_free:.1f}GB free"
                    )
                if gpu_temp and gpu_temp > 80:
                    response.add_warning(f"High GPU temperature: {gpu_temp}°C")

            else:
                response.gpu_status = {
                    "available": False,
                    "reason": "CUDA not available",
                    "status": "unavailable",
                }
                response.checks["gpu_available"] = False

                if config.DEVICE == "cuda":
                    response.add_warning("CUDA device configured but not available")

        except Exception as exc:
            logger.error(f"GPU check failed: {exc}")
            response.checks["gpu_available"] = False
            response.add_warning(f"GPU check failed: {str(exc)}")

    # === File System Check ===
    try:
        # Check critical paths
        critical_paths = [
            config.OUTPUT_PATH,
            config.get_model_path(),
            Path(config.LOG_FILE).parent,
        ]

        path_status = {}
        all_paths_ok = True

        for path in critical_paths:
            path_obj = Path(path)
            exists = path_obj.exists()
            writable = (
                path_obj.is_dir() and os.access(path, os.W_OK) if exists else False
            )

            path_status[str(path)] = {
                "exists": exists,
                "writable": writable,
                "status": "ok" if exists and writable else "error",
            }

            if not exists or not writable:
                all_paths_ok = False
                response.add_warning(f"Path issue: {path}")

        response.checks["file_system"] = all_paths_ok

        if verbose:
            response.system_status["file_paths"] = path_status

    except Exception as exc:
        logger.error(f"File system check failed: {exc}")
        response.checks["file_system"] = False
        response.add_warning("File system check failed")

    # === Model Availability Check (if enabled) ===
    if check_models:
        try:
            # Check primary model path
            model_path = Path(config.get_model_path())
            model_exists = model_path.exists()

            response.model_status = {
                "primary_model": config.PRIMARY_MODEL,
                "model_path": str(model_path),
                "model_exists": model_exists,
                "status": "available" if model_exists else "missing",
            }

            response.checks["models_available"] = model_exists

            if not model_exists:
                response.add_warning(f"Primary model not found: {model_path}")
                response.add_recommendation(
                    "Run model installation script: python scripts/install_models.py"
                )

        except Exception as exc:
            logger.error(f"Model check failed: {exc}")
            response.checks["models_available"] = False
            response.add_warning("Model availability check failed")

    # === Final Status Check ===
    check_results = list(response.checks.values())

    if all(check_results):
        response.status = "healthy"
    elif any(check_results):
        response.status = "degraded"
    else:
        response.status = "unhealthy"

    # === Recommendations ===
    if (
        response.gpu_status
        and response.gpu_status.get("memory", {}).get("usage_percent", 0) > 70
    ):
        response.add_recommendation(
            "Consider enabling CPU offload to reduce GPU memory usage"
        )

    if response.system_status.get("memory", {}).get("used_percent", 0) > 80:
        response.add_recommendation(
            "Consider reducing batch size or enabling memory optimization"
        )

    # === Timing Information ===
    check_time = time.time() - start_time
    response.service_info["health_check_time"] = round(check_time, 3)

    logger.info(
        f"Health check completed: {response.status}",
        extra={
            "status": response.status,
            "checks_passed": sum(response.checks.values()),
            "total_checks": len(response.checks),
            "warnings": len(response.warnings),
            "check_time": check_time,
        },
    )

    return response


@router.get("/health/quick")
async def quick_health_check() -> Dict[str, Any]:
    """Quick health check endpoint for basic service status"""
    try:
        # Basic service status
        cuda_available = torch.cuda.is_available()

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "cuda_available": cuda_available,
            "device": settings.DEVICE,
            "service": "running",
        }

    except Exception as exc:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(exc),
            "service": "error",
        }


@router.get("/health/detailed")
async def detailed_health_check(
    config: Settings = Depends(get_settings),
) -> HealthCheckResponse:
    """Detailed health check endpoint for comprehensive diagnostics"""

    return await health_check(
        check_gpu=True, check_models=True, verbose=True, config=config
    )
