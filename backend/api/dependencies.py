# backend/api/dependencies.py
"""
FastAPI Dependencies

Provides shared dependencies for API endpoints including:
- Authentication and authorization
- Request validation and rate limiting
- Common data access patterns
"""

import time
from typing import Optional
from fastapi import Depends, HTTPException, Request
from backend.core.sd_pipeline import sd_manager
import logging

logger = logging.getLogger(__name__)


async def get_sd_manager():
    """
    Dependency to ensure SD manager is available

    Returns:
        SD manager instance

    Raises:
        HTTPException: If manager is not properly initialized
    """
    if sd_manager.pipeline is None:
        # Attempt to load default model
        if not sd_manager.load_pipeline():
            raise HTTPException(
                status_code=503,
                detail="Stable Diffusion model not loaded. Please check model configuration.",
            )

    return sd_manager


async def log_request_time(request: Request):
    """
    Dependency to log request processing time

    Args:
        request: FastAPI request object
    """
    start_time = time.time()

    def log_time():
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")

    request.state.log_time = log_time
    return start_time
