# app/schemas/responses.py
"""
SD Multi-Modal Platform - API Response Schemas
This module defines the response schemas for the text-to-image generation API.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, field_validator, Field, conlist, Field
from pydantic_settings import BaseSettings
from datetime import datetime
import time


class ImageMetadata(BaseModel):
    """Metadata for generated images"""

    # General metadata
    seed: int = Field(..., description="Random seed used for generation")
    prompt: str = Field(..., description="Prompt used for image generation")
    negative_prompt: str = Field(default="", description="Negative prompt")

    # Model and generation details
    model: str = Field(..., description="Model used for generation")
    model_hash: Optional[str] = Field(default=None, description="Model hash")

    # Generation parameters
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    steps: int = Field(..., description="Inference steps")
    cfg_scale: float = Field(..., description="CFG scale (guidance scale)")
    sampler: Optional[str] = Field(default=None, description="Sampler name")
    scheduler: Optional[str] = Field(default=None, description="Scheduler name")

    # Generation statistics
    generation_time: float = Field(
        ..., description="Time taken for generation (seconds)"
    )
    timestamp: float = Field(
        default_factory=time.time, description="Generation timestamp"
    )

    # Hardware and performance
    device: str = Field(..., description="Device used for generation (CPU/GPU)")
    vram_used: Optional[str] = Field(
        default=None, description="VRAM used (if applicable)"
    )
    peak_memory: Optional[float] = Field(
        default=None, description="Peak memory usage during generation (in GB)"
    )

    # Output details
    filename: str = Field(..., description="Generated image filename")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    file_format: str = Field(default="PNG", description="File format (e.g., PNG, JPEG)")

    # Platform metadata
    platform_version: str = Field(
        default="1.0.0-phase1", description="Platform version"
    )

    def to_dict(self) -> dict:
        """Convert metadata to dictionary format for serialization"""
        return self.model_dump(exclude_none=True)

    def get_filename_suffix(self) -> str:
        """Generate a filename suffix based on metadata"""
        return f"_{self.seed}_{self.steps}s_{self.cfg_scale}cfg"


class GeneratedImage(BaseModel):
    """Represents a generated image with metadata and file information"""

    # Image file information
    url: str = Field(..., description="Image URL or path")
    filename: str = Field(..., description="Image filename")
    local_path: Optional[str] = Field(
        default=None, description="Local file path (if applicable)"
    )

    # Image data
    base64_data: Optional[str] = Field(
        default=None, description="Base64 encoded image data (if applicable)"
    )

    # Metadata
    metadata: ImageMetadata = Field(..., description="Image metadata")

    # Image dimensions
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    seed: int = Field(..., description="Random seed used for generation")

    def get_public_url(self, base_url: str = "") -> str:
        """Generate a public URL for the image"""
        if self.url.startswith("http"):
            return self.url
        return f"{base_url.rstrip('/')}/{self.url.lstrip('/')}"


class Txt2ImgResponse(BaseModel):
    """Response schema for text-to-image generation API"""

    # Response metadata
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(
        default="Generation completed successfully", description="Response message"
    )
    timestamp: float = Field(
        default_factory=time.time, description="Response timestamp"
    )

    # Request metadata
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional data related to the request"
    )

    # Generated images
    images: List[GeneratedImage] = Field(
        default_factory=list, description="List of generated images"
    )

    # Task and performance metadata
    task_id: str = Field(..., description="Unique identifier for the generation task")
    total_images: int = Field(..., description="Total number of images generated")

    # Performance metrics
    total_time: float = Field(
        ..., description="Total time taken for the generation (seconds)"
    )
    model_load_time: Optional[float] = Field(
        default=None, description="Time taken to load the model (if applicable)"
    )
    generation_time: float = Field(
        ..., description="Time taken for image generation (seconds)"
    )

    # Model and device information
    model_used: str = Field(..., description="Model used for generation")
    device_used: str = Field(..., description="Device used for generation (CPU/GPU)")

    # 錯誤資訊 (成功時為空)
    error: Optional[str] = Field(
        default=None, description="Wrong message if generation failed"
    )
    error_code: Optional[str] = Field(
        default=None, description="Error code if applicable"
    )

    @classmethod
    def success_response(
        cls,
        images: List[GeneratedImage],
        task_id: str,
        total_time: float,
        generation_time: float,
        model_used: str,
        device_used: str,
        model_load_time: Optional[float] = None,
    ) -> "Txt2ImgResponse":
        """Constuct a successful response"""
        return cls(
            success=True,
            message="Generation completed  successfully",
            images=images,
            task_id=task_id,
            total_images=len(images),
            total_time=total_time,
            model_load_time=model_load_time,
            generation_time=generation_time,
            model_used=model_used,
            device_used=device_used,
            data={
                "batch_size": len(images),
                "average_time_per_image": generation_time / max(len(images), 1),
            },
        )

    @classmethod
    def error_response(
        cls,
        error_message: str,
        task_id: str,
        error_code: Optional[str] = None,
        model_used: Optional[str] = None,
        device_used: Optional[str] = None,
    ) -> "Txt2ImgResponse":
        """Construct an error response"""
        return cls(
            success=False,
            message="Generation failed",
            task_id=task_id,
            total_images=0,
            total_time=0.0,
            generation_time=0.0,
            model_used=model_used or "unknown",
            device_used=device_used or "unknown",
            error=error_message,
            error_code=error_code,
            images=[],
        )


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint"""

    status: str = Field(..., description="Service state: healthy/unhealthy/degraded")
    timestamp: float = Field(
        default_factory=time.time, description="Health check timestamp"
    )

    # Service information
    service_info: Dict[str, Any] = Field(
        default_factory=dict, description="Service metadata and configuration"
    )

    # System status
    system_status: Dict[str, Any] = Field(
        default_factory=dict, description="System status information"
    )

    # GPU status
    gpu_status: Optional[Dict[str, Any]] = Field(
        default=None, description="GPU status information (if applicable)"
    )

    # Model status
    model_status: Optional[Dict[str, Any]] = Field(
        default=None, description="Model status information (if applicable)"
    )

    # Checks and diagnostics
    checks: Dict[str, bool] = Field(
        default_factory=dict, description="Health checks and diagnostics"
    )

    # Warnings and recommendations
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )

    def is_healthy(self) -> bool:
        """Check if the service is healthy"""
        return self.status == "healthy"

    def add_warning(self, warning: str):
        """Add a warning to the health check response"""
        if warning not in self.warnings:
            self.warnings.append(warning)

    def add_recommendation(self, recommendation: str):
        """Add a recommendation to the health check response"""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)


class ModelInfo(BaseModel):
    """Model information schema"""

    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Model name")
    type: str = Field(
        ..., description="Model type (e.g., 'text-to-image', 'image-to-image')"
    )
    path: str = Field(..., description="Path to the model file or directory")

    # Model status
    loaded: bool = Field(default=False, description="Whether the model is loaded")
    loading: bool = Field(
        default=False, description="Whether the model is currently loading"
    )
    active: bool = Field(
        default=False, description="Whether the model is currently active"
    )

    # Model metadata
    memory_usage: Optional[float] = Field(
        default=None, description="Memory usage (in GB)"
    )
    load_time: Optional[float] = Field(
        default=None, description="Time taken to load the model (in seconds)"
    )

    # Model capabilities
    supports_controlnet: bool = Field(
        default=False, description="Whether the model supports ControlNet"
    )
    supports_lora: bool = Field(
        default=False, description="Whether the model supports LoRA"
    )
    supports_img2img: bool = Field(
        default=False,
        description="Whether the model supports image-to-image generation",
    )

    # Recommended settings
    recommended_resolution: List[int] = Field(
        default_factory=lambda: [1024, 1024], description="Recommended image resolution"
    )
    recommended_steps: int = Field(
        default=25, description="Recommended inference steps"
    )
    recommended_cfg: float = Field(default=7.5, description="Recommended CFG")


class ModelListResponse(BaseModel):
    """Response schema for model listing API"""

    success: bool = Field(
        default=True, description="Whether the request was successful"
    )
    models: List[ModelInfo] = Field(..., description="Working models list")
    active_model: Optional[str] = Field(
        default=None, description="Active model ID (if any)"
    )
    total_models: int = Field(..., description="Total number of models available")
    loaded_models: int = Field(..., description="Number of models currently loaded")

    # System performance metrics
    memory_info: Optional[Dict[str, Any]] = Field(
        default=None, description="System memory information (if available)"
    )

    def get_active_model(self) -> Optional[ModelInfo]:
        """Retrieve the currently active model from the list of models."""
        for model in self.models:
            if model.active:
                return model
        return None


# Error response schema
class ErrorResponse(BaseModel):
    """Schema for error responses in the API"""

    success: bool = Field(
        default=False, description="Whether the request was successful"
    )
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(
        default=None, description="Error code (if applicable)"
    )
    message: Optional[str] = Field(
        default=None, description="Additional message or details about the error"
    )

    # Request metadata
    request_id: Optional[str] = Field(
        default=None, description="Unique identifier for the request"
    )
    timestamp: float = Field(
        default_factory=time.time, description="Error response timestamp"
    )

    # Debug information
    debug_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Debug information (if applicable)"
    )

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
        debug_mode: bool = False,
    ) -> "ErrorResponse":
        """Create an ErrorResponse from an exception."""
        response = cls(error=str(exc), error_code=error_code, request_id=request_id)

        if debug_mode:
            import traceback

            response.debug_info = {
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }

        return response
