# app/schemas/responses.py
"""
Response schemas for SD Multi-Modal Platform API endpoints.
Defines Pydantic models for response validation and documentation.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, field_validator, Field, conlist, Field
from pydantic_settings import BaseSettings
from datetime import datetime
import time


class ImageInfo(BaseModel):
    """Information about a generated image."""

    index: int = Field(description="Image index in batch")
    width: int = Field(description="Image width in pixels")
    height: int = Field(description="Image height in pixels")
    mode: str = Field(description="Image color mode (e.g., 'RGB')")
    file_path: Optional[str] = Field(
        default=None, description="Relative path to saved image file"
    )
    file_size_bytes: Optional[int] = Field(
        default=None, description="File size in bytes"
    )
    base64: Optional[str] = Field(default=None, description="Base64 encoded image data")


class ModelInfo(BaseModel):
    """Information about a model."""

    model_id: str = Field(description="Model identifier")
    model_name: str = Field(description="Human-readable model name")
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


class GenerationParams(BaseModel):
    """Parameters used for generation."""

    prompt: str = Field(description="Text prompt used")
    negative_prompt: str = Field(description="Negative prompt used")
    width: int = Field(description="Image width")
    height: int = Field(description="Image height")
    steps: int = Field(description="Number of inference steps")
    cfg_scale: float = Field(description="CFG scale value")
    seed: int = Field(description="Random seed used")


class GenerationResults(BaseModel):
    """Results of image generation."""

    num_images: int = Field(description="Number of images generated")
    images: List[ImageInfo] = Field(description="Information about generated images")
    generation_time: float = Field(description="Time spent generating images (seconds)")
    total_time: float = Field(description="Total request processing time (seconds)")
    vram_used_gb: float = Field(description="VRAM usage in GB")


class Txt2ImgResponseData(BaseModel):
    """Data payload for txt2img response."""

    task_id: str = Field(description="Unique task identifier")
    model_used: ModelInfo = Field(description="Model information")
    generation_params: GenerationParams = Field(description="Generation parameters")
    results: GenerationResults = Field(description="Generation results")
    optimization_info: Dict[str, Any] = Field(description="Optimization details")


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
    """Response schema for text-to-image generation."""

    success: bool = Field(description="Whether generation was successful")
    task_id: str = Field(description="Unique task identifier")
    message: str = Field(description="Status message")
    data: Optional[Dict[str, Any]] = Field(
        description="Generation results and metadata"
    )
    request_id: str = Field(description="Request tracking ID")
    timestamp: float = Field(description="Response timestamp")


# Error response schema
class ErrorResponse(BaseModel):
    """Standard error response format."""

    success: bool = Field(
        default=False, description="Whether the request was successful"
    )
    error: str = Field(..., description="Error message")
    status_code: int = Field(description="HTTP status code")
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


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(description="Health status (healthy/degraded/unhealthy)")
    timestamp: float = Field(description="Check timestamp")
    service: Dict[str, str] = Field(description="Service information")
    system: Dict[str, Any] = Field(description="System information")
    gpu: Optional[Dict[str, Any]] = Field(default=None, description="GPU information")


class ModelStatusResponse(BaseModel):
    """Model status response schema."""

    service: str = Field(description="Service name")
    endpoint: str = Field(description="API endpoint")
    model_manager: Dict[str, Any] = Field(description="Model manager status")
    supported_parameters: Dict[str, str] = Field(description="Supported parameters")
    available_models: Dict[str, Any] = Field(description="Available models information")


class ModelListResponse(BaseModel):
    """Model list response schema."""

    available_models: Dict[str, Any] = Field(description="Detailed model information")
    currently_loaded: Optional[str] = Field(description="Currently loaded model ID")
    total_models: int = Field(description="Total number of models")
    installed_models: int = Field(description="Number of installed models")


class ModelSwitchResponse(BaseModel):
    """Model switch response schema."""

    success: bool = Field(description="Whether switch was successful")
    message: str = Field(description="Switch result message")
    previous_model: Optional[str] = Field(description="Previous model ID")
    current_model: str = Field(description="Current model ID")
    switch_time: float = Field(description="Time taken to switch models")


class APIInfoResponse(BaseModel):
    """API information response schema."""

    api: Dict[str, str] = Field(description="API metadata")
    endpoints: Dict[str, str] = Field(description="Available endpoints")
    model_manager: Dict[str, Any] = Field(description="Model manager status")
    configuration: Dict[str, Any] = Field(description="System configuration")
    optimizations: Dict[str, Any] = Field(description="Performance optimizations")
