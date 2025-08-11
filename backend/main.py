# backend/main.py
"""
FastAPI Application Entry Point

Main application configuration with:
- CORS middleware for cross-origin requests
- Error handling and logging setup
- Health check endpoints
- Static file serving for generated images
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exception_handlers import http_exception_handler
import uvicorn
import torch

from backend.api.v1.router import api_router
from backend.config.settings import Settings
from backend.schemas.responses import HealthCheckResponse, ErrorResponse
from backend.core.sd_pipeline import sd_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(Settings.log_dir) / "app.log"),
    ],
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="SD Multimodal Platform API",
    description="Stable Diffusion API with multiple interface support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests
    """
    start_time = datetime.now()

    response = await call_next(request)

    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"path": str(request.url.path)},
        ).dict(),
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom HTTP exception handler with structured responses
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP{exc.status_code}",
            message=exc.detail,
            details={"path": str(request.url.path)},
        ).dict(),
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Service health check endpoint

    Returns system status, model loading state, and resource information.
    """
    try:
        # Check if SD model is loaded
        model_loaded = sd_manager.pipeline is not None

        # Memory information
        memory_info = None
        if torch.cuda.is_available():
            memory_info = {
                "cuda_available": True,
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "cached_gb": torch.cuda.memory_reserved() / (1024**3),
            }
        else:
            memory_info = {"cuda_available": False}

        return HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            model_loaded=model_loaded,
            device=Settings.device,
            memory_info=memory_info,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Root endpoint
@app.get("/")
async def root():
    """
    API root endpoint with basic information
    """
    return {
        "service": "SD Multimodal Platform API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


# Include API routes
app.include_router(api_router, prefix=Settings.api_prefix)

# Mount static files for serving generated images
app.mount(
    "/api/v1/images",
    StaticFiles(directory=Settings.output_dir),
    name="generated_images",
)


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Application startup initialization
    """
    logger.info("🚀 Starting SD Multimodal Platform API")
    logger.info(f"Device: {Settings.device}")
    logger.info(f"Model path: {Settings.sd_model_path}")

    # Pre-load SD model if available
    try:
        if Path(Settings.sd_model_path).exists():
            logger.info("Pre-loading Stable Diffusion model...")
            if sd_manager.load_pipeline():
                logger.info("✅ Model loaded successfully")
            else:
                logger.warning("⚠️ Model loading failed, will load on first request")
        else:
            logger.info("📥 Model not found locally, will download on first request")
    except Exception as e:
        logger.error(f"Startup model loading failed: {e}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown cleanup
    """
    logger.info("🛑 Shutting down SD Multimodal Platform API")

    # Clean up resources
    if sd_manager.pipeline is not None:
        sd_manager.unload_pipeline()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("✅ Cleanup completed")


# Development server entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Settings.host,
        port=Settings.port,
        reload=Settings.reload,
        log_level="info",
    )
