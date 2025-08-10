# app/main.py
"""
SD Multi-Modal Platform - FastAPI Main Application
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
import torch
import platform

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings, validate_phase1_setup
from app.api.v1 import txt2img, health
from utils.logging_utils import setup_logging
from services.models.sd_models import get_model_manager


# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown tasks"""
    # === Initialization on startup ===
    logger.info("🚀 Starting SD Multi-Modal Platform...")

    # Validate Phase 1 setup
    if not validate_phase1_setup():
        logger.error("❌ Phase 1 setup validation failed")
        raise RuntimeError("Configuration validation failed")

    # Preload models if needed
    try:
        model_manager = get_model_manager()
        await model_manager.warm_up()
        logger.info("✅ Model preloading completed")
    except Exception as e:
        logger.warning(f"⚠️  Model preloading failed: {e}")
        # Phase 1 does not require preloading, so we can continue

    logger.info(f"✅ Server ready on {settings.HOST}:{settings.PORT}")

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
    description="Production-ready multi-model text-to-image platform with intelligent routing",
    version="1.0.0-phase1",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    lifespan=lifespan,
)


# === Middleware Configuration ===

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Request Logging Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Middleware to log requests and responses"""
    start_time = time.time()

    # Generate a unique request ID
    request_id = f"req_{int(time.time() * 1000)}"

    # Record the request
    logger.info(
        f"🔄 [{request_id}] {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
        },
    )

    # Process the request
    try:
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log the response
        logger.info(
            f"✅ [{request_id}] {response.status_code} - {process_time:.3f}s",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time,
            },
        )

        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        return response

    except Exception as exc:
        # Log the exception with request ID
        process_time = time.time() - start_time
        logger.error(
            f"❌ [{request_id}] Error: {str(exc)} - {process_time:.3f}s",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "process_time": process_time,
            },
            exc_info=True,
        )

        # Return a JSON error response
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
            headers={"X-Request-ID": request_id},
        )


# === Global Exception Handlers ===


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Process HTTP exceptions globally"""
    request_id = getattr(request.state, "request_id", "unknown")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Process validation errors globally"""
    request_id = getattr(request.state, "request_id", "unknown")

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "message": str(exc),
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


# === API Routes Setup ===

# Health check
app.include_router(health.router, tags=["Health"])

# V1 API Routes
app.include_router(
    txt2img.router, prefix=settings.API_PREFIX, tags=["Text-to-Image Generation"]
)

# === Static Files Setup ===
if Path(settings.OUTPUT_PATH).exists():
    app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_PATH), name="outputs")


# === Root and Info Endpoints ===


@app.get("/")
async def root():
    """Root endpoint for basic information"""
    return {
        "message": "SD Multi-Modal Platform API",
        "version": "1.0.0-phase1",
        "phase": "Phase 1: MVP Backend + Basic txt2img",
        "docs": f"{settings.API_PREFIX}/docs",
        "health": "/health",
        "primary_model": settings.PRIMARY_MODEL,
        "device": settings.DEVICE,
        "api_endpoints": [f"{settings.API_PREFIX}/txt2img", "/health", "/outputs"],
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
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD_ON_CHANGE,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.ENABLE_REQUEST_LOGGING,
    )
