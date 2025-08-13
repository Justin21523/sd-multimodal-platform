# app/main.py
"""
astAPI Main Application - Phase 3 Integration
Updated to include real model management and txt2img generation.
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
from app.api.v1 import txt2img, health
from services.models.sd_models import get_model_manager
from utils.logging_utils import setup_logging, get_request_logger


# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Check if running in minimal mode
MINIMAL_MODE = os.getenv("MINIMAL_MODE", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown tasks"""
    # === Initialization on startup ===
    startup_start = time.time()
    logger.info("🚀 Starting SD Multi-Modal Platform...")

    if MINIMAL_MODE:
        logger.info("🔧 Running in MINIMAL MODE - skipping model initialization")

    try:
        # Validate configuration
        logger.info("Validating configuration...")
        settings.ensure_directories()

        # Initialize model manager only if not in minimal mode
        model_manager_ready = False
        if not MINIMAL_MODE:
            logger.info("Initializing model manager...")
            model_manager = get_model_manager()

            # Try to initialize with primary model
            try:
                initialization_success = await model_manager.initialize()

                if not initialization_success:
                    logger.warning("⚠️  Model manager initialization failed")
                    logger.warning(
                        "The API will start but txt2img endpoints will be unavailable"
                    )
                    logger.warning(
                        "Run 'python scripts/install_models.py' to download models"
                    )
                else:
                    logger.info(
                        f"✅ Model manager initialized with: {model_manager.current_model_id}"
                    )
                    model_manager_ready = True

            except Exception as e:
                logger.error(f"Model initialization error: {e}")
                logger.warning("Continuing startup without model initialization...")
        else:
            logger.info("✅ Model initialization skipped (minimal mode)")

        startup_time = time.time() - startup_start
        mode_info = "MINIMAL MODE" if MINIMAL_MODE else "FULL MODE"
        logger.info(
            f"🎉 Application startup completed in {startup_time:.2f}s ({mode_info})"
        )

        # Store startup info for health checks
        app.state.startup_time = startup_time
        app.state.model_manager_ready = model_manager_ready
        app.state.minimal_mode = MINIMAL_MODE

        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        # Continue startup even if model init fails
        app.state.startup_time = time.time() - startup_start
        app.state.model_manager_ready = False
        app.state.minimal_mode = MINIMAL_MODE
        yield

    finally:
        # Cleanup on shutdown
        logger.info("🔄 Shutting down application...")

        try:
            if not MINIMAL_MODE:
                model_manager = get_model_manager()
                if model_manager.is_initialized:
                    await model_manager.cleanup()
                    logger.info("✅ Model manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        logger.info("👋 Application shutdown completed")


# === FastAPI Application Instance ===
app = FastAPI(
    title="SD Multi-Modal Platform",
    description="Production-ready multi-model text-to-image platform with intelligent routing",
    version="1.0.0-phase3",
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


# === Root and Info Endpoints ===
# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "SD Multi-Modal Platform",
        "version": "1.0.0-phase3",
        "phase": "Phase 3: Model Management & Real Generation",
        "mode": "minimal" if MINIMAL_MODE else "full",
        "api_docs": f"{settings.API_PREFIX}/docs",
        "health_check": f"{settings.API_PREFIX}/health",
        "status": "operational",
        "features": [
            (
                "Real text-to-image generation"
                if not MINIMAL_MODE
                else "API structure testing"
            ),
            "Multiple model support (SDXL, SD1.5, SD2.1)",
            (
                "Model switching and management"
                if not MINIMAL_MODE
                else "Model management (disabled)"
            ),
            "RTX 5080 optimizations",
            "Memory management",
            "Request tracking",
            "Structured logging",
        ],
    }


# Additional API information endpoint
@app.get(f"{settings.API_PREFIX}/info")
async def api_info():
    """Get comprehensive API information."""
    model_manager = get_model_manager()

    # Basic info that works in minimal mode
    info = {
        "api": {
            "prefix": settings.API_PREFIX,
            "version": "v1",
            "docs_url": f"{settings.API_PREFIX}/docs",
        },
        "endpoints": {
            "health": f"{settings.API_PREFIX}/health",
            "txt2img": f"{settings.API_PREFIX}/txt2img",
            "txt2img_status": f"{settings.API_PREFIX}/txt2img/status",
            "models": f"{settings.API_PREFIX}/txt2img/models",
        },
        "model_manager": {
            "initialized": model_manager.is_initialized,
            "current_model": model_manager.current_model_id,
            "startup_time": model_manager.startup_time,
        },
        "configuration": {
            "device": settings.DEVICE,
            "primary_model": settings.PRIMARY_MODEL,
            "max_batch_size": settings.MAX_BATCH_SIZE,
            "default_width": settings.DEFAULT_WIDTH,
            "default_height": settings.DEFAULT_HEIGHT,
        },
        "optimizations": {
            "use_sdpa": settings.USE_SDPA,
            "enable_xformers": settings.ENABLE_XFORMERS,
            "attention_slicing": settings.USE_ATTENTION_SLICING,
            "cpu_offload": settings.ENABLE_CPU_OFFLOAD,
        },
        "mode": {
            "minimal_mode": MINIMAL_MODE,
            "model_manager_ready": getattr(app.state, "model_manager_ready", False),
        },
    }

    # Add model manager info if available
    if not MINIMAL_MODE:
        try:
            model_manager = get_model_manager()
            info["model_manager"] = {
                "initialized": model_manager.is_initialized,
                "current_model": model_manager.current_model_id,
                "startup_time": model_manager.startup_time,
            }
        except Exception as e:
            info["model_manager"] = {"error": str(e), "initialized": False}
    else:
        info["model_manager"] = {"disabled": "Running in minimal mode"}

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
