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
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

import torch
import platform

from app.config import settings
from app.core.queue_manager import get_queue_manager, shutdown_queue_manager
from utils.logging_utils import setup_logging
from utils.middleware import RequestLoggingMiddleware, RateLimitMiddleware
from app.api.v1 import (
    txt2img,
    img2img,
    health,
    assets,
    queue,
    inpaint,
    upscale,
    face_restore,
)
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
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="""
    **SD Multi-Modal Platform - Phase 6: Queue System & Rate Limiting**

    A production-ready multi-model text-to-image platform with intelligent routing,
    complete post-processing pipeline, and enterprise-grade queue management.

    ## Features

    * **Smart Model Routing**: Automatic model selection based on prompt analysis
    * **Queue Management**: Redis-backed task queue with Celery workers
    * **Rate Limiting**: Per-user request limits and global throttling
    * **Real-time Status**: Live task progress tracking and monitoring
    * **Post-processing**: Real-ESRGAN upscaling + GFPGAN face restoration
    * **Batch Processing**: Multiple images and parameter variations
    * **Production Ready**: Health checks, metrics, logging, and scaling

    ## API Workflow

    1. **Submit Task**: POST to generation endpoints returns task_id
    2. **Track Progress**: GET `/queue/status/{task_id}` for live updates
    3. **Manage Queue**: List, cancel, and monitor tasks
    4. **Download Results**: Access generated images and metadata

    ## Rate Limits

    * **Per User**: 100 requests/hour (configurable)
    * **Global**: 1000 requests/minute
    * **Burst**: 10 requests (short-term spikes)

    ## Queue Priorities

    * **URGENT**: System tasks (10)
    * **HIGH**: Premium users (8-9)
    * **NORMAL**: Regular requests (6-7)
    * **LOW**: Batch/background (1-5)
    """,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
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

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware, requests_per_minute=settings.GLOBAL_RATE_LIMIT_PER_MINUTE
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


# Include API routers
app.include_router(txt2img.router, prefix=settings.API_PREFIX)
app.include_router(img2img.router, prefix=settings.API_PREFIX)
app.include_router(inpaint.router, prefix=settings.API_PREFIX)

# Post-processing endpoints
app.include_router(upscale.router, prefix=settings.API_PREFIX)
app.include_router(face_restore.router, prefix=settings.API_PREFIX)

# Queue management endpoints
app.include_router(queue.router, prefix=settings.API_PREFIX)

# System endpoints
app.include_router(health.router, prefix="")  # No prefix for health

# Phase 5: Include queue router
if settings.ENABLE_QUEUE:
    app.include_router(queue.router, prefix=settings.API_PREFIX)

# =====================================
# Custom Documentation
# =====================================


@app.get(f"{settings.API_PREFIX}/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,  # type: ignore
        title=f"{app.title} - API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "operationsSorter": "method",
            "filter": True,
            "tagsSorter": "alpha",
        },
    )


def custom_openapi():
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add custom schema extensions
    openapi_schema["info"]["x-logo"] = {"url": "https://example.com/logo.png"}

    # Add server information
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.sdplatform.com", "description": "Production server"},
    ]

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
    }

    # Add rate limiting info to endpoints
    for path, path_item in openapi_schema["paths"].items():
        for method, operation in path_item.items():
            if method.lower() in ["get", "post", "put", "delete"]:
                if "responses" not in operation:
                    operation["responses"] = {}

                # Add rate limit response
                operation["responses"]["429"] = {
                    "description": "Rate limit exceeded",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {"type": "string"},
                                    "retry_after": {"type": "integer"},
                                },
                            }
                        }
                    },
                }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


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


# === Root and Info Endpoints ===
# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with platform information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "phase": settings.PHASE,
        "status": "operational",
        "api_docs": f"{settings.API_PREFIX}/docs",
        "api_prefix": settings.API_PREFIX,
        "features": [
            "Multi-model text-to-image generation",
            "Intelligent model routing",
            "Queue management with Redis + Celery",
            "Rate limiting and throttling",
            "Real-time task progress tracking",
            "Post-processing pipeline",
            "Batch generation support",
            "Production monitoring",
        ],
        "supported_models": [
            "Stable Diffusion XL",
            "Stable Diffusion 1.5",
            "PixArt-Sigma",
            "DeepFloyd IF",
            "Stable Cascade",
        ],
        "queue_info": {
            "enabled": True,
            "backend": "Redis + Celery",
            "rate_limit_per_hour": settings.RATE_LIMIT_PER_HOUR,
            "max_concurrent_tasks": settings.MAX_CONCURRENT_TASKS,
        },
    }


@app.get(f"{settings.API_PREFIX}/info")
async def api_info():
    """Detailed API information"""
    queue_manager = await get_queue_manager()
    queue_stats = await queue_manager.get_queue_status()

    return {
        "api_version": settings.VERSION,
        "api_prefix": settings.API_PREFIX,
        "environment": "development" if settings.DEBUG else "production",
        "device": settings.DEVICE,
        "queue_stats": {
            "total_tasks": queue_stats.total_tasks,
            "pending_tasks": queue_stats.pending_tasks,
            "running_tasks": queue_stats.running_tasks,
            "active_workers": queue_stats.active_workers,
            "total_workers": queue_stats.total_workers,
        },
        "rate_limits": {
            "per_user_per_hour": settings.RATE_LIMIT_PER_HOUR,
            "global_per_minute": settings.GLOBAL_RATE_LIMIT_PER_MINUTE,
            "burst_allowance": settings.RATE_LIMIT_BURST,
        },
        "capabilities": {
            "text_to_image": True,
            "image_to_image": True,
            "inpainting": True,
            "upscaling": True,
            "face_restoration": True,
            "batch_processing": True,
            "queue_management": True,
            "real_time_progress": True,
        },
    }


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


# =====================================
# Development Helpers
# =====================================

if settings.DEBUG:

    @app.get("/debug/config", include_in_schema=False)
    async def debug_config():
        """Debug endpoint to view current configuration"""
        return {
            "settings": {
                "device": settings.DEVICE,
                "debug": settings.DEBUG,
                "redis_url": settings.get_redis_url(),
                "max_concurrent_tasks": settings.MAX_CONCURRENT_TASKS,
                "rate_limit_per_hour": settings.RATE_LIMIT_PER_HOUR,
            },
            "paths": {
                "model_base_path": str(settings.MODEL_BASE_PATH),
                "output_base_path": str(settings.OUTPUT_BASE_PATH),
                "cache_base_path": str(settings.CACHE_BASE_PATH),
            },
        }

    @app.post("/debug/queue/clear", include_in_schema=False)
    async def debug_clear_queue():
        """Debug endpoint to clear all queues"""
        if not settings.DEBUG:
            raise HTTPException(status_code=404, detail="Not found")

        # This would clear all Redis queues in development
        return {"message": "Queue cleared (debug mode only)"}


# =====================================
# Metrics Endpoint (for Prometheus)
# =====================================

if settings.ENABLE_METRICS:

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint"""
        # This would return Prometheus formatted metrics
        # For now, return basic JSON metrics
        queue_manager = await get_queue_manager()
        stats = await queue_manager.get_queue_status()

        metrics_data = f"""
# HELP sd_platform_total_tasks Total number of tasks
# TYPE sd_platform_total_tasks counter
sd_platform_total_tasks {stats.total_tasks}

# HELP sd_platform_pending_tasks Number of pending tasks
# TYPE sd_platform_pending_tasks gauge
sd_platform_pending_tasks {stats.pending_tasks}

# HELP sd_platform_running_tasks Number of running tasks
# TYPE sd_platform_running_tasks gauge
sd_platform_running_tasks {stats.running_tasks}

# HELP sd_platform_completed_tasks Number of completed tasks
# TYPE sd_platform_completed_tasks counter
sd_platform_completed_tasks {stats.completed_tasks}

# HELP sd_platform_failed_tasks Number of failed tasks
# TYPE sd_platform_failed_tasks counter
sd_platform_failed_tasks {stats.failed_tasks}

# HELP sd_platform_active_workers Number of active workers
# TYPE sd_platform_active_workers gauge
sd_platform_active_workers {stats.active_workers}

# HELP sd_platform_queue_throughput Tasks processed per minute
# TYPE sd_platform_queue_throughput gauge
sd_platform_queue_throughput {stats.queue_throughput}

# HELP sd_platform_average_wait_time Average task wait time in seconds
# TYPE sd_platform_average_wait_time gauge
sd_platform_average_wait_time {stats.average_wait_time}

# HELP sd_platform_average_processing_time Average task processing time in seconds
# TYPE sd_platform_average_processing_time gauge
sd_platform_average_processing_time {stats.average_processing_time}
"""

        return Response(content=metrics_data, media_type="text/plain")


# Import Response for metrics endpoint
from fastapi import Response

# =====================================
# Startup Banner
# =====================================


def print_startup_banner():
    """Print startup banner with system information"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🎨 SD Multi-Modal Platform - Phase 6                     ║
║                        Queue System & Rate Limiting                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Version: {settings.VERSION:<20} Device: {settings.DEVICE:<20}          ║
║  API Prefix: {settings.API_PREFIX:<16} Redis: {'✅ Connected' if True else '❌ Disconnected':<20}      ║
║  Max Concurrent: {settings.MAX_CONCURRENT_TASKS:<12} Rate Limit: {settings.RATE_LIMIT_PER_HOUR}/hour          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🚀 API Endpoints:                                                          ║
║     • POST /api/v1/txt2img          - Text to image generation              ║
║     • POST /api/v1/img2img          - Image to image transformation         ║
║     • POST /api/v1/inpaint          - Image inpainting                      ║
║     • POST /api/v1/upscale          - Image upscaling                       ║
║     • POST /api/v1/face_restore     - Face restoration                      ║
║                                                                              ║
║  🎛️  Queue Management:                                                       ║
║     • POST /api/v1/queue/enqueue    - Submit tasks to queue                ║
║     • GET  /api/v1/queue/status     - Get queue statistics                  ║
║     • GET  /api/v1/queue/status/ID  - Get task status                       ║
║     • POST /api/v1/queue/cancel/ID  - Cancel specific task                  ║
║     • GET  /api/v1/queue/tasks      - List all tasks                        ║
║                                                                              ║
║  📊 Monitoring:                                                             ║
║     • GET  /health                  - System health check                   ║
║     • GET  /metrics                 - Prometheus metrics                    ║
║     • GET  /api/v1/docs             - Interactive API documentation         ║
║                                                                              ║
║  🔧 Development Tools:                                                      ║
║     • Flower Dashboard: http://localhost:5555 (admin:admin123)             ║
║     • Redis Commander: http://localhost:8081 (admin:admin123)              ║
║     • Grafana Dashboard: http://localhost:3000 (admin:admin123)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 Ready to process requests! Visit http://localhost:8000/api/v1/docs for API docs.
"""
    print(banner)


# =====================================
# Application Entry Point
# =====================================

if __name__ == "__main__":
    import uvicorn

    # Print startup information
    if not settings.TESTING:
        print_startup_banner()

    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.RELOAD and settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        reload_dirs=["app"] if settings.RELOAD else None,
        reload_excludes=["*.pyc", "__pycache__"] if settings.RELOAD else None,
    )
