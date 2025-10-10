# app/schemas/responses.py
"""
Response schemas for SD Multi-Modal Platform API endpoints.
Response schemas for SD Multi-Modal Platform APIs
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, field_validator, Field, conlist, Field
from pydantic_settings import BaseSettings
from datetime import datetime
import time
from .common import BaseResponse  # type: ignore


class BaseResponse(BaseModel):
    """Base response model for all API endpoints"""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Human-readable message")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp")

    class Config:
        arbitrary_types_allowed = True


# Error response schema
class ErrorResponse(BaseResponse):
    """Standard error response format."""

    success: bool = Field(
        default=False, description="Whether the request was successful"
    )
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(
        default=None, description="Error code for programmatic handling"
    )
    details: Optional[Any] = None
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


class LivenessResponse(BaseModel):
    """Liveness probe response"""

    ok: bool
    timestamp: datetime
    message: str


class ReadinessResponse(BaseModel):
    """Readiness probe response"""

    ok: bool
    timestamp: datetime
    message: str
    warehouse_accessible: bool
    system_healthy: bool
    device_available: bool
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseResponse):
    """Health check response model"""

    """Health check response"""
    status: str = "healthy"
    version: str = ""
    cache_initialized: bool = False

    class HealthData(BaseModel):
        status: str = Field(
            ..., description="Health status: healthy, degraded, unhealthy"
        )
        service: Dict[str, str] = Field(..., description="Service information")
        system: Dict[str, Any] = Field(..., description="System information")
        gpu: Optional[Dict[str, str]] = Field(
            default=None, description="GPU information"
        )
        capabilities: Dict[str, bool] = Field(..., description="Available capabilities")


class ModelInfo(BaseModel):
    """Model information schema"""

    model_id: str = Field(description="Model identifier")
    name: str = Field(description="Human-readable model name")
    type: str = Field(
        ..., description="Model type (e.g., 'text-to-image', 'image-to-image')"
    )
    capabilities: List[str] = Field(..., description="Supported capabilities")
    vram_requirement: str = Field(..., description="VRAM requirement")
    strengths: List[str] = Field(default_factory=list, description="Model strengths")
    recommended_for: List[str] = Field(
        default_factory=list, description="Recommended use cases"
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


class GenerationMetadata(BaseModel):
    """Generation metadata schema"""

    task_id: str = Field(..., description="Unique task identifier")
    prompt: str = Field(..., description="Original prompt")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    model_used: str = Field(..., description="Model used for generation")
    seed: Optional[int] = Field(default=None, description="Random seed used")
    generation_params: Dict[str, Any] = Field(..., description="Generation parameters")
    processing_time: float = Field(..., description="Total processing time in seconds")
    vram_usage: str = Field(..., description="VRAM usage during generation")
    image_count: int = Field(..., description="Number of images generated")
    created_at: float = Field(
        default_factory=time.time, description="Creation timestamp"
    )


class GenerationResults(BaseModel):
    """Generation results schema"""

    images: List[str] = Field(..., description="Generated image URLs or paths")
    metadata: GenerationMetadata = Field(..., description="Generation metadata")
    processing_time: Dict[str, float] = Field(
        ..., description="Breakdown of processing times"
    )
    num_images: int = Field(description="Number of images generated")
    generation_time: float = Field(description="Time spent generating images (seconds)")
    total_time: float = Field(description="Total request processing time (seconds)")
    vram_used_gb: float = Field(description="VRAM usage in GB")

    class ProcessingTime(BaseModel):
        total: float = Field(..., description="Total processing time")
        preprocessing: float = Field(default=0.0, description="Preprocessing time")
        generation: float = Field(..., description="Generation time")
        postprocessing: float = Field(default=0.0, description="Postprocessing time")


class GenerationResponse(BaseResponse):
    """Standard generation response"""

    success: bool = Field(default=True)
    message: str = Field(default="Generation completed successfully")
    data: Dict[str, Any] = Field(..., description="Generation results and metadata")


class Txt2ImgResponse(GenerationResponse):
    """Text-to-image generation response"""

    image_url: Optional[str] = None
    task_id: Optional[str] = Field(..., description="Task identifier")
    metadata: Optional[Dict[str, Any]] = None
    request_id: str = Field(description="Request tracking ID")

    class Txt2ImgData(BaseModel):
        metadata: GenerationMetadata = Field(..., description="Generation metadata")
        timestamp: float = Field(description="Response timestamp")
        model_info: Dict[str, str] = Field(..., description="Model information")


class Img2ImgResponse(GenerationResponse):
    """Image-to-image generation response"""

    """Image to image generation response"""
    image_url: Optional[str] = None
    task_id: Optional[str] = Field(..., description="Task identifier")
    metadata: Optional[Dict[str, Any]] = None

    class Img2ImgData(BaseModel):
        images: List[str] = Field(..., description="Generated image paths")
        metadata: GenerationMetadata = Field(..., description="Generation metadata")
        processing_time: Dict[str, float] = Field(
            ..., description="Processing time breakdown"
        )
        controlnet_info: Optional[Dict[str, Any]] = Field(
            default=None, description="ControlNet information"
        )
        model_info: Dict[str, str] = Field(..., description="Model information")


class CaptionResponse(BaseResponse):
    """Image captioning response"""

    caption: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VQAResponse(BaseResponse):
    """Visual question answering response"""

    answer: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class QueueStatusResponse(BaseResponse):
    """Queue task status response"""

    task_id: str
    status: str
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QueueListResponse(BaseResponse):
    """Queue task list response"""

    tasks: List[Dict[str, Any]]
    total: int


class InpaintResponse(GenerationResponse):
    """Inpainting generation response"""

    class InpaintData(BaseModel):
        task_id: str = Field(..., description="Task identifier")
        images: List[str] = Field(..., description="Generated image paths")
        metadata: GenerationMetadata = Field(..., description="Generation metadata")
        inpaint_info: Dict[str, Any] = Field(
            ..., description="Inpainting-specific information"
        )
        processing_time: Dict[str, float] = Field(
            ..., description="Processing time breakdown"
        )


class ModelListResponse(BaseResponse):
    """Model list response"""

    success: bool = Field(default=True)
    data: Dict[str, Any] = Field(..., description="Available models information")
    models: List[Dict[str, Any]]
    categories: List[str]

    class ModelListData(BaseModel):
        available_models: List[ModelInfo] = Field(
            ..., description="List of available models"
        )
        current_model: Optional[str] = Field(
            default=None, description="Currently loaded model"
        )
        total_models: int = Field(..., description="Total number of available models")
        installed_models: int = Field(description="Number of installed models")


class ModelLoadResponse(BaseResponse):
    """Model load response"""

    model_name: str
    loaded: bool
    device: str
    dtype: str


class ModelStatusResponse(BaseResponse):
    """Model status response"""

    success: bool = Field(default=True)
    data: Dict[str, Any] = Field(..., description="Model status information")
    service: str = Field(description="Service name")
    endpoint: str = Field(description="API endpoint")

    class ModelStatusData(BaseModel):
        current_model: Optional[str] = Field(
            default=None, description="Currently loaded model"
        )
        is_loaded: bool = Field(..., description="Whether a model is loaded")
        vram_usage: str = Field(..., description="Current VRAM usage")
        supported_tasks: List[str] = Field(
            ..., description="Supported generation tasks"
        )
        model_info: Optional[ModelInfo] = Field(
            default=None, description="Current model details"
        )
        model_manager: Dict[str, Any] = Field(description="Model manager status")
        supported_parameters: Dict[str, str] = Field(description="Supported parameters")
        available_models: Dict[str, Any] = Field(
            description="Available models information"
        )


class AssetInfo(BaseModel):
    """Asset information schema"""

    asset_id: str = Field(..., description="Asset unique identifier")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="File storage path")
    thumbnail_path: Optional[str] = Field(default=None, description="Thumbnail path")
    category: str = Field(..., description="Asset category")
    tags: List[str] = Field(default_factory=list, description="Asset tags")
    description: str = Field(default="", description="Asset description")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: Optional[str] = Field(default=None, description="MIME type")
    image_info: Dict[str, Any] = Field(
        default_factory=dict, description="Image properties"
    )
    created_at: float = Field(..., description="Creation timestamp")
    updated_at: float = Field(..., description="Last update timestamp")
    usage_count: int = Field(default=0, description="Usage count")
    last_used: Optional[float] = Field(default=None, description="Last used timestamp")


class AssetUploadResponse(BaseResponse):
    """Asset upload response"""

    success: bool = Field(default=True)
    data: Dict[str, Any] = Field(..., description="Upload results")

    class AssetUploadData(BaseModel):
        uploaded_assets: List[Dict[str, Any]] = Field(
            ..., description="Successfully uploaded assets"
        )
        failed_uploads: List[Dict[str, str]] = Field(
            ..., description="Failed uploads with errors"
        )
        summary: Dict[str, int] = Field(..., description="Upload summary statistics")
        processing_time: float = Field(..., description="Total processing time")


class AssetListResponse(BaseResponse):
    """Asset list response"""

    success: bool = Field(default=True)
    data: Dict[str, Any] = Field(..., description="Asset list data")

    class AssetListData(BaseModel):
        assets: List[AssetInfo] = Field(..., description="List of assets")
        pagination: Dict[str, int] = Field(..., description="Pagination information")
        filters: Dict[str, Any] = Field(..., description="Applied filters")
        total_count: int = Field(..., description="Total assets matching filters")


class AssetResponse(BaseResponse):
    """Single asset response"""

    success: bool = Field(default=True)
    data: AssetInfo = Field(..., description="Asset information")


class ServiceStatusResponse(BaseResponse):
    """Service status response"""

    success: bool = Field(default=True)
    data: Dict[str, Any] = Field(..., description="Service status data")

    class ServiceStatusData(BaseModel):
        service_available: bool = Field(..., description="Service availability")
        current_model: Optional[str] = Field(default=None, description="Current model")
        supported_models: List[str] = Field(..., description="Supported models")
        vram_usage: str = Field(..., description="VRAM usage")
        capabilities: Dict[str, bool] = Field(..., description="Service capabilities")
        performance_stats: Optional[Dict[str, float]] = Field(
            default=None, description="Performance statistics"
        )


class ControlNetStatusResponse(BaseResponse):
    """ControlNet status response"""

    success: bool = Field(default=True)
    data: Dict[str, Any] = Field(..., description="ControlNet status data")

    class ControlNetStatusData(BaseModel):
        loaded_processors: List[str] = Field(
            ..., description="Loaded ControlNet processors"
        )
        supported_types: List[str] = Field(
            ..., description="Supported ControlNet types"
        )
        pipeline_loaded: bool = Field(
            ..., description="Whether ControlNet pipeline is loaded"
        )
        total_vram_usage: str = Field(..., description="Total VRAM usage")
        available_models: Dict[str, str] = Field(
            ..., description="Available ControlNet models"
        )


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


# Helper function to create standard responses
def create_success_response(
    message: str = "Operation completed successfully",
    data: Any = None,
    response_class: type = BaseResponse,
) -> Dict[str, Any]:
    """Create a standard success response"""
    response_data = {"success": True, "message": message, "timestamp": time.time()}

    if data is not None:
        response_data["data"] = data

    return response_data


def create_error_response(
    message: str,
    error_details: Optional[str] = None,
    error_code: Optional[str] = None,
    request_id: Optional[str] = None,
    status_code: int = 500,
) -> Dict[str, Any]:  # type: ignore[return]
    """Create a standard error response"""
    response_data = {
        "success": False,
        "message": message,
        "error": error_details or message,
        "timestamp": time.time(),
    }

    if error_code:
        response_data["error_code"] = error_code

    if request_id:
        response_data["request_id"] = request_id
