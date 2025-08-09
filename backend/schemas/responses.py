# backend/schemas/responses.py
"""
API Response Models

Defines consistent response structures for all endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ImageData(BaseModel):
    """Individual image data in response"""

    filename: str = Field(description="Generated filename")
    url: str = Field(description="URL to access the image")
    width: int = Field(description="Image width in pixels")
    height: int = Field(description="Image height in pixels")
    seed: Optional[int] = Field(description="Seed used for generation")


class Text2ImageResponse(BaseModel):
    """Response for text-to-image generation"""

    success: bool = Field(description="Generation success status")
    message: str = Field(description="Response message")
    images: List[ImageData] = Field(description="Generated images data")
    generation_time: float = Field(description="Time taken for generation (seconds)")
    parameters: Dict[str, Any] = Field(description="Parameters used for generation")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class ModelInfoResponse(BaseModel):
    """Response with current model information"""

    success: bool = Field(description="Request success status")
    model_info: Dict[str, Any] = Field(description="Current pipeline information")
    available_models: Dict[str, str] = Field(description="Available model options")


class HealthCheckResponse(BaseModel):
    """Health check response"""

    status: str = Field(description="Service health status")
    version: str = Field(description="API version")
    model_loaded: bool = Field(description="Whether SD model is loaded")
    device: str = Field(description="Compute device being used")
    memory_info: Optional[Dict[str, Any]] = Field(
        description="Memory usage information"
    )


class ErrorResponse(BaseModel):
    """Error response model"""

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error type")
    message: str = Field(description="Detailed error message")
    details: Optional[Dict[str, Any]] = Field(description="Additional error details")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Error timestamp"
    )


class StatusResponse(BaseModel):
    """General status response"""

    success: bool = Field(description="Operation success status")
    message: str = Field(description="Status message")
    data: Optional[Dict[str, Any]] = Field(description="Additional data")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )
