# app/main.py
"""
astAPI Main Application - Phase 3 Integration
FastAPI main application with Phase 5 queue and post-processing integration
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any
import os

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import torch
import platform

from app.config import settings
from app.api.v1 import txt2img, img2img, health, assets, queue
from services.models.sd_models import get_model_manager
from services.queue.task_manager import get_task_manager
from utils.logging_utils import setup_logging, get_request_logger


# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Check if running in minimal mode
MINIMAL_MODE = os.getenv("MINIMAL_MODE", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown tasks"""
    logger = get_request_logger("startup")

    # === Initialization on startup ===
    startup_start = time.time()
    logger.info("🚀 Starting SD Multi-Modal Platform - Phase 5")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Queue enabled: {settings.ENABLE_QUEUE}")

    if MINIMAL_MODE:
        logger.info("🔧 Running in MINIMAL MODE - skipping model initialization")

    try:
        # Initialize task manager and check Redis connection
        if settings.ENABLE_QUEUE:
            logger.info("📋 Initializing task queue system...")
            task_manager = get_task_manager()

            # Test Redis connection
            try:
                stats = await task_manager.get_queue_stats()
                if stats.get("redis_connected"):
                    logger.info("✅ Redis connection successful")
                else:
                    logger.warning("⚠️ Redis connection failed - queue disabled")
            except Exception as e:
                logger.warning(f"⚠️ Queue system initialization failed: {str(e)}")

        # Initialize model manager (from Phase 4)
        if not settings.MINIMAL_MODE:
            logger.info("🤖 Initializing model manager...")
            from services.models.sd_models import get_model_manager

            model_manager = get_model_manager()

            try:
                success = await model_manager.initialize()
                if success:
                    logger.info(
                        f"✅ Model manager initialized: {model_manager.current_model}"
                    )
                else:
                    logger.warning(
                        "⚠️ Model initialization failed - running in limited mode"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Model initialization error: {str(e)}")

        # Initialize post-processing pipeline
        if settings.ENABLE_POSTPROCESS:
            logger.info("🎨 Initializing post-processing pipeline...")
            try:
                from services.postprocess.pipeline_manager import get_pipeline_manager

                pipeline_manager = get_pipeline_manager()
                logger.info("✅ Post-processing pipeline ready")
            except Exception as e:
                logger.warning(f"⚠️ Post-processing initialization failed: {str(e)}")

        logger.info("✅ Application startup completed")

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("🔄 Shutting down application...")

    try:
        # Cleanup model manager
        if not settings.MINIMAL_MODE:
            logger.info("🧹 Cleaning up model manager...")
            from services.models.sd_models import get_model_manager

            model_manager = get_model_manager()
            await model_manager.cleanup()

        # Cleanup post-processing models
        if settings.ENABLE_POSTPROCESS:
            logger.info("🧹 Cleaning up post-processing pipeline...")
            from services.postprocess.pipeline_manager import get_pipeline_manager

            pipeline_manager = get_pipeline_manager()
            await pipeline_manager._unload_all_models()

        # Cleanup old tasks if queue enabled
        if settings.ENABLE_QUEUE:
            logger.info("🧹 Cleaning up old tasks...")
            task_manager = get_task_manager()
            cleaned_count = await task_manager.cleanup_old_tasks()
            logger.info(f"🗑️ Cleaned up {cleaned_count} old tasks")

        logger.info("✅ Application shutdown completed")

    except Exception as e:
        logger.error(f"❌ Shutdown error: {str(e)}")


# === FastAPI Application Instance ===
app = FastAPI(
    title="SD Multi-Modal Platform",
    description="Advanced multi-model text-to-image platform with queue system and post-processing",
    version="1.0.0-phase5",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    lifespan=lifespan,
)


# === Middleware Configuration ===

# CORS Middleware - must be added before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)


# Request Logging Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Add request ID, timing, and structured logging to all requests."""

    # Generate a unique request ID
    request_id = f"req_{int(time.time() * 1000)}"
    # Add request ID to request state for downstream use
    request.state.request_id = request_id

    # Start timing
    start_time = time.time()

    # Get request logger with ID
    req_logger = get_request_logger(request_id)
    # Log request start
    req_logger.info(
        f"🔄 Request started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
        },
    )

    try:
        # Process the request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        # Add minimal mode header if applicable
        if MINIMAL_MODE:
            response.headers["X-Minimal-Mode"] = "true"

        # Log successful response
        req_logger.info(
            f"✅ Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time,
            },
        )

        return response

    except Exception as exc:
        # Calculate processing time even for errors
        process_time = time.time() - start_time

        # Log error
        req_logger.error(
            f"❌ Request failed: {str(exc)}",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "process_time": process_time,
            },
            exc_info=True,
        )

        # Return a structured error response
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "message": (
                    str(exc) if settings.DEBUG else "An unexpected error occurred"
                ),
                "request_id": request_id,
                "timestamp": time.time(),
                "minimal_mode": MINIMAL_MODE,
            },
            headers={
                "X-Request-ID": request_id,
                "X-Process-Time": f"{process_time:.3f}s",
            },
        )


# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with structured response."""

    request_id = getattr(request.state, "request_id", "unknown")

    # Log the exception
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )

    # Return structured error response
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
            "timestamp": time.time(),
            "minimal_mode": MINIMAL_MODE,
        },
        headers={"X:Request-ID": request_id},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": f"Validation error: {str(exc)}",
            "status_code": 422,
            "request_id": request_id,
            "timestamp": time.time(),
            "minimal_mode": MINIMAL_MODE,
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unexpected errors."""

    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={"request_id": request_id, "error_type": type(exc).__name__},
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "status_code": 500,
            "request_id": request_id,
            "timestamp": time.time(),
            "minimal_mode": MINIMAL_MODE,
        },
        headers={"X-Request-ID": request_id},
    )


# Include API routers
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(txt2img.router, prefix=settings.API_PREFIX)
app.include_router(img2img.router, prefix=settings.API_PREFIX)
app.include_router(assets.router, prefix=settings.API_PREFIX)

# Phase 5: Include queue router
if settings.ENABLE_QUEUE:
    app.include_router(queue.router, prefix=settings.API_PREFIX)


# === Root and Info Endpoints ===
# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "SD Multi-Modal Platform",
        "version": "1.0.0-phase5",
        "mode": "minimal" if MINIMAL_MODE else "full",
        "api_docs": f"{settings.API_PREFIX}/docs",
        "health_check": f"{settings.API_PREFIX}/health",
        "status": "running",
        "features": {
            "generation": True,
            "img2img": True,
            "controlnet": True,
            "assets": True,
            "queue": settings.ENABLE_QUEUE,
            "postprocess": settings.ENABLE_POSTPROCESS,
        },
        "api_docs": "/docs",
        "health_check": f"{settings.API_PREFIX}/health",
    }


# Enhanced info endpoint
@app.get(f"{settings.API_PREFIX}/info")
async def system_info(request: Request):
    """Detailed system information"""
    import torch
    import platform

    info = {
        "platform": {
            "system": platform.system(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
        },
        "hardware": {
            "device": settings.DEVICE,
            "cuda_available": torch.cuda.is_available(),
        },
        "configuration": {
            "minimal_mode": settings.MINIMAL_MODE,
            "primary_model": settings.PRIMARY_MODEL,
            "max_batch_size": settings.MAX_BATCH_SIZE,
            "queue_enabled": settings.ENABLE_QUEUE,
            "postprocess_enabled": settings.ENABLE_POSTPROCESS,
        },
        "api": {
            "prefix": settings.API_PREFIX,
            "version": "v1",
            "request_id": getattr(request.state, "request_id", None),
        },
    }

    # Add GPU information if CUDA available
    if torch.cuda.is_available():
        info["hardware"]["gpu_name"] = torch.cuda.get_device_name(0)
        info["hardware"][
            "gpu_memory"
        ] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"

    # Add queue stats if enabled
    if settings.ENABLE_QUEUE:
        try:
            task_manager = get_task_manager()
            queue_stats = await task_manager.get_queue_stats()
            info["queue"] = queue_stats
        except Exception:
            info["queue"] = {"status": "unavailable"}

    return info


@app.get("/info")
async def server_info():
    """Server information endpoint"""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": settings.DEVICE,
        "primary_model": settings.PRIMARY_MODEL,
        "output_path": settings.OUTPUT_PATH,
        "debug_mode": settings.DEBUG,
    }

    # Include GPU details if available
    if torch.cuda.is_available():
        info.update(
            {
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            }
        )

    return info


if __name__ == "__main__":
    # Check if minimal mode is requested
    if "--minimal" in sys.argv:
        os.environ["MINIMAL_MODE"] = "true"
        print("🔧 Starting in MINIMAL MODE")

    # Development server
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,  # 0.0.0.0
        port=settings.PORT,  # 8000
        reload=True,  # True
        access_log=False,  # Disable access logs
        log_level=settings.LOG_LEVEL.lower(),  # info
    )
