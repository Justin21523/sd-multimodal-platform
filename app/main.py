# app/main.py
"""
FastAPI main application with middleware, error handling, and API documentation.
Phase 2: Backend Framework & Basic API Services
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
import torch
import platform
import uuid
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.v1 import txt2img, health
from utils.logging_utils import setup_logging, get_request_logger
from services.models.sd_models import get_model_manager


# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown tasks"""
    # === Initialization on startup ===
    logger.info("🚀 Starting SD Multi-Modal Platform...")

    try:
        # Ensure required directories exist
        settings.ensure_directories()

        # Basic system validation
        logger.info(f"Device: {settings.DEVICE}")
        logger.info(f"API Prefix: {settings.API_PREFIX}")
        logger.info(f"Max Workers: {settings.MAX_WORKERS}")

        # Future: Model manager warm-up will go here
        logger.info("✅ Application startup completed")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield

    # === Cleanup on shutdown ===
    logger.info("🛑 Shutting down SD Multi-Modal Platform...")

    # Clean up models if needed
    try:
        model_manager = get_model_manager()
        await model_manager.cleanup()
        logger.info("✅ Model cleanup completed")
    except Exception as e:
        logger.error(f"❌ Model cleanup failed: {e}")


# === FastAPI Application Instance ===
app = FastAPI(
    title="SD Multi-Modal Platform",
    description="Production-ready multi-model text-to-image generation platform",
    version="1.0.0-phase2",
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
            f"❌ Request failed",
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
                    str(exc) if settings.DEBUG_MODE else "An unexpected error occurred"
                ),
                "request_id": request_id,
                "timestamp": time.time(),
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
        },
        headers={"X:Request-ID": request_id},
    )


@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handle validation errors with 422 status."""

    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(f"Validation error: {str(exc)}", extra={"request_id": request_id})

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": f"Validation error: {str(exc)}",
            "status_code": 422,
            "request_id": request_id,
            "timestamp": time.time(),
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
        },
        headers={"X-Request-ID": request_id},
    )


# === API Routes Setup (Include routers) ===
#  health checkers
app.include_router(health.router, prefix=settings.API_PREFIX, tags=["Health Check"])


# === Root and Info Endpoints ===
# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "message": "SD Multi-Modal Platform API",
        "version": "1.0.0-phase2",
        "phase": "Phase 2: Backend Framework & Basic API Services",
        "docs": f"{settings.API_PREFIX}/docs",
        "health": f"{settings.API_PREFIX}/health",
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
        "debug_mode": settings.DEBUG_MODE,
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

    # Development server
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,  # 0.0.0.0
        port=settings.PORT,  # 8000
        reload=settings.RELOAD_ON_CHANGE,  # True
        log_level=settings.LOG_LEVEL.lower(),  # info
    )
