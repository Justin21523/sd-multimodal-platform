"""
FastAPI application entrypoint for sd-multimodal-platform.

Goals for this repo:
- Backend stays bootable even when optional components (Redis/Celery/ControlNet) are missing.
- All storage paths follow ~/Desktop/data_model_structure.md (models/caches under /mnt/c, outputs under /mnt/data).
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on sys.path for older absolute imports in the repo.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.queue_manager import shutdown_queue_manager
from app.shared_cache import shared_cache  # noqa: F401  (side-effect: set cache env vars)
from utils.logging_utils import setup_logging

from app.api.middleware.auth import AuthMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.v1 import assets, face_restore, health, history, img2img, models, queue, txt2img, upscale

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not getattr(settings, "TESTING", False):
        setup_logging(log_level=settings.LOG_LEVEL)
        logger.info(
            "Starting API",
            extra={
                "environment": settings.ENVIRONMENT,
                "minimal_mode": settings.MINIMAL_MODE,
                "api_prefix": settings.API_PREFIX,
            },
        )

    # Best-effort model warmup so sync endpoints work out-of-the-box.
    # Keep the API bootable even when models aren't installed yet.
    if not settings.MINIMAL_MODE and not getattr(settings, "TESTING", False):
        try:
            from services.models.sd_models import get_model_manager

            manager = get_model_manager()
            if not getattr(manager, "is_initialized", False):
                ok = await manager.initialize(settings.PRIMARY_MODEL)
                if not ok:
                    logger.warning(
                        "ModelManager not initialized (models missing?) — generation endpoints will return 503 until installed.",
                        extra={"models_path": settings.MODELS_PATH, "primary_model": settings.PRIMARY_MODEL},
                    )
        except Exception as exc:
            logger.warning(f"Model warmup skipped: {exc}")

    yield

    try:
        await shutdown_queue_manager()
    except Exception:
        pass


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    lifespan=lifespan,
)

# -----------------------------
# Middleware
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)

if settings.ENABLE_RATE_LIMITING:
    app.add_middleware(RateLimitMiddleware)

app.add_middleware(AuthMiddleware)

# -----------------------------
# Routers
# -----------------------------

# Health is useful both at root (docker/ops) and under the API prefix (clients).
app.include_router(health.router)
app.include_router(health.router, prefix=settings.API_PREFIX)

app.include_router(txt2img.router, prefix=settings.API_PREFIX)
app.include_router(img2img.router, prefix=settings.API_PREFIX)
app.include_router(upscale.router, prefix=settings.API_PREFIX)
app.include_router(face_restore.router, prefix=settings.API_PREFIX)
app.include_router(queue.router, prefix=settings.API_PREFIX)
app.include_router(assets.router, prefix=settings.API_PREFIX)
app.include_router(history.router, prefix=settings.API_PREFIX)
app.include_router(models.router, prefix=settings.API_PREFIX)

# -----------------------------
# Static files
# -----------------------------

# These mount points are convenience for UI clients. `check_dir=False` avoids
# boot failures when the target directory doesn't exist yet.
app.mount(
    "/outputs",
    StaticFiles(directory=str(settings.OUTPUT_PATH), check_dir=False),
    name="outputs",
)
app.mount(
    "/assets",
    StaticFiles(directory=str(settings.ASSETS_PATH), check_dir=False),
    name="assets",
)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": app.docs_url,
        "openapi": app.openapi_url,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
