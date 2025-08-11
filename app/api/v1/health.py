# app/api/v1/health.py
"""
Health check API endpoints for system monitoring and status reporting.
Phase 2: Backend Framework & Basic API Services
"""
import os

import logging
import time
import platform
import psutil
import sys
from pathlib import Path
from typing import Dict, Any, Literal

from fastapi import APIRouter, Request, Depends, Query
import torch
import pynvml as nvml

from app.config import Settings, settings, get_settings
from app.schemas.requests import HealthCheckRequest
from app.schemas.responses import HealthCheckResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    summary="System Health Check",
    description="Comprehensive health check including hardware and service status",
    response_model=HealthCheckResponse,
)
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Comprehensive system health check with hardware detection and service status.

    Returns:
        - System information (platform, Python version)
        - Hardware status (CUDA availability, GPU info)
        - Service configuration
        - Performance metrics
    """

    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()

    try:
        # Basic system info
        system_info = {
            "platform": platform.platform(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "device": settings.DEVICE,
            "cuda_available": torch.cuda.is_available(),
        }

        # GPU information if CUDA is available
        gpu_info = {}
        if torch.cuda.is_available():
            try:
                gpu_info = {
                    "gpu_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                }

                # Memory usage if possible
                if hasattr(torch.cuda, "memory_allocated"):
                    gpu_info.update(
                        {
                            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
                            "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}GB",
                        }
                    )

            except Exception as e:
                logger.warning(f"Failed to get GPU info: {e}")
                gpu_info = {"error": "Failed to retrieve GPU information"}

        # Service configuration
        service_info = {
            "name": "SD Multi-Modal Platform",
            "version": "1.0.0-phase2",
            "phase": "Phase 2: Backend Framework & Basic API Services",
            "api_prefix": settings.API_PREFIX,
            "max_workers": settings.MAX_WORKERS,
            "primary_model": settings.PRIMARY_MODEL,
            "torch_dtype": settings.TORCH_DTYPE,
        }

        # Determine overall health status
        health_status: Literal["healthy", "degraded", "unhealthy"] = "healthy"

        # Check for potential issues
        issues = []

        if not torch.cuda.is_available() and settings.DEVICE == "cuda":
            health_status = "degraded"
            issues.append("CUDA not available but configured for CUDA device")

        if gpu_info.get("error"):
            health_status = "degraded"
            issues.append("GPU information retrieval failed")

        # Performance metrics
        response_time = time.time() - start_time

        response_data = {
            "status": health_status,
            "timestamp": time.time(),
            "response_time": f"{response_time:.3f}s",
            "request_id": request_id,
            "service": service_info,
            "system": system_info,
            "issues": issues,
        }

        # Add GPU info if available
        if gpu_info:
            response_data["gpu"] = gpu_info

        logger.info(
            "Health check completed",
            extra={
                "request_id": request_id,
                "status": health_status,
                "response_time": response_time,
                "cuda_available": torch.cuda.is_available(),
            },
        )

        return response_data

    except Exception as e:
        logger.error(
            f"Health check failed: {e}", extra={"request_id": request_id}, exc_info=True
        )

        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "request_id": request_id,
            "error": str(e),
            "service": {"name": "SD Multi-Modal Platform", "version": "1.0.0-phase2"},
        }


@router.get(
    "/health/simple",
    summary="Simple Health Check",
    description="Lightweight health check for load balancers",
)
async def simple_health_check() -> Dict[str, str]:
    """
    Simple health check endpoint for load balancers and monitoring systems.
    Returns minimal response for high-frequency checks.
    """
    return {
        "status": "ok",
        "service": "sd-multimodal-platform",
        "timestamp": str(int(time.time())),
    }


@router.get(
    "/health/detailed",
    summary="Detailed System Information",
    description="Extended system information for debugging and monitoring",
)
async def detailed_health_check(request: Request) -> Dict[str, Any]:
    """
    Detailed health check with extended system information for debugging.
    Includes environment variables, model paths, and directory status.
    """

    request_id = getattr(request.state, "request_id", "unknown")

    try:
        # Get basic health info
        basic_health = await health_check(request)

        # Add detailed configuration
        config_info = {
            "environment": {
                "device": settings.DEVICE,
                "torch_dtype": settings.TORCH_DTYPE,
                "default_width": settings.DEFAULT_WIDTH,
                "default_height": settings.DEFAULT_HEIGHT,
                "default_steps": settings.DEFAULT_STEPS,
                "default_cfg": settings.DEFAULT_CFG,
                "max_batch_size": settings.MAX_BATCH_SIZE,
                "output_path": str(settings.OUTPUT_PATH),
                "enable_xformers": settings.ENABLE_XFORMERS,
                "use_attention_slicing": settings.USE_ATTENTION_SLICING,
            },
            "model_paths": {
                "sd_model_path": str(settings.SD_MODEL_PATH),
                "sdxl_model_path": str(settings.SDXL_MODEL_PATH),
                "controlnet_path": str(settings.CONTROLNET_PATH),
                "lora_path": str(settings.LORA_PATH),
                "vae_path": str(settings.VAE_PATH),
            },
        }

        # Check directory existence
        directory_status = {}
        for name, path in config_info["model_paths"].items():

            directory_status[name] = {
                "exists": Path(path).exists(),
                "is_directory": Path(path).is_dir() if Path(path).exists() else False,
                "path": path,
            }

        # Combine all information
        detailed_response = {
            **basic_health,
            "configuration": config_info,
            "directory_status": directory_status,
        }

        logger.info("Detailed health check completed", extra={"request_id": request_id})

        return detailed_response

    except Exception as e:
        logger.error(
            f"Detailed health check failed: {e}",
            extra={"request_id": request_id},
            exc_info=True,
        )

        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "request_id": request_id,
            "error": f"Detailed health check failed: {str(e)}",
        }
