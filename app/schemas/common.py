"""
Common Pydantic schemas
"""

from pydantic import BaseModel
from typing import Optional, Any, List
from datetime import datetime


class BaseResponse(BaseModel):
    """Base response model"""

    success: bool = True
    message: str = "Success"
    timestamp: datetime = None  # type: ignore

    class Config:
        arbitrary_types_allowed = True


class ErrorResponse(BaseResponse):
    """Error response model"""

    success: bool = False
    error_code: str = ""
    details: Optional[Any] = None


class HealthResponse(BaseResponse):
    """Health check response"""

    status: str = "healthy"
    version: str = ""
    cache_initialized: bool = False
