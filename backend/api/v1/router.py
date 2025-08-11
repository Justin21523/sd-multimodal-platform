# backend/api/v1/router.py
"""
API Router Configuration

Centralizes all API endpoints and provides versioned routing.
"""

from fastapi import APIRouter
from backend.api.v1.endpoints import txt2img, models

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(txt2img.router)
api_router.include_router(models.router)
