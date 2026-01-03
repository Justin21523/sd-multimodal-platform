# app/schemas/requests.py
"""
Request schemas for SD Multi-Modal Platform API endpoints.
Defines Pydantic models for request validation and documentation.
"""

from typing import Optional, Dict, Any, List, Literal

import base64
import re
import uuid

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from app.config import settings


# Extend existing Txt2ImgRequest
class ControlNetConfig(BaseModel):
    """ControlNet configuration for conditional generation"""

    type: Literal["canny", "depth", "openpose", "scribble", "mlsd", "normal"] = Field(
        ..., description="ControlNet processor type"
    )
    image: Optional[str] = Field(
        default=None, description="Base64 encoded condition image (legacy)"
    )
    asset_id: Optional[str] = Field(
        default=None, description="Asset ID for condition image (preferred)"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path to condition image (restricted to ASSETS_PATH/OUTPUT_PATH)",
    )
    preprocess: bool = Field(
        default=True,
        description="Whether to auto-preprocess the control image (requires controlnet-aux/annotators).",
    )
    strength: float = Field(
        default=1.0, ge=0.0, le=2.0, description="ControlNet influence strength"
    )
    guidance_start: float = Field(default=0.0, ge=0.0, le=1.0)
    guidance_end: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("image")
    @classmethod
    def validate_base64_image(cls, v: Optional[str]) -> Optional[str]:
        """Validate base64 image format and size"""
        if v is None:
            return None
        try:
            # Remove data URL prefix if present
            if v.startswith("data:image"):
                v = v.split(",", 1)[1]

            # Decode and validate
            image_data = base64.b64decode(v)
            if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("Image size exceeds 10MB limit")

            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {str(e)}")

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        try:
            uuid.UUID(str(v))
        except Exception as e:
            raise ValueError(f"Invalid asset_id (expected UUID): {e}")
        return str(v)

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        from app.schemas.queue_requests import _resolve_allowed_image_path

        return _resolve_allowed_image_path(v)

    @model_validator(mode="after")
    def validate_image_source(self) -> "ControlNetConfig":
        provided = [
            self.image is not None,
            self.asset_id is not None,
            self.image_path is not None,
        ]
        if sum(provided) != 1:
            raise ValueError(
                "ControlNet requires exactly one of: image (base64), asset_id, image_path"
            )
        return self


class GenerationParams(BaseModel):
    """Basic generation parameters for text-to-image requests"""

    width: int = Field(default=1024, ge=256, le=2048, description="image width")
    height: int = Field(default=1024, ge=256, le=2048, description="image height")
    num_inference_steps: int = Field(
        default=25, ge=10, le=100, description="inference steps"
    )
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(
        default=None, description="random seed for reproducibility"
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=4,  # Will be validated against settings.MAX_BATCH_SIZE
        description="Number of images to generate",
    )

    @field_validator("width", "height", mode="before")
    @classmethod
    def validate_dimensions(cls, v):
        if v is not None and v % 8 != 0:
            # Round to nearest multiple of 8
            v = ((v + 7) // 8) * 8
        return v

    @field_validator("seed", mode="before")
    @classmethod
    def handle_negative_seed(cls, v):
        """Convert -1 to None for random seed."""
        return None if v == -1 else v


class Txt2ImgRequest(BaseModel):
    """Request schema for text-to-image generation."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)

    model_id: Optional[str] = Field(default=None, description="Model ID (optional)")
    user_id: Optional[str] = Field(default=None, description="User ID (optional)")

    width: int = Field(default=settings.DEFAULT_WIDTH, ge=256, le=2048)
    height: int = Field(default=settings.DEFAULT_HEIGHT, ge=256, le=2048)
    num_inference_steps: int = Field(
        default=settings.DEFAULT_STEPS,
        ge=1,
        le=settings.MAX_STEPS,
        validation_alias=AliasChoices("num_inference_steps", "steps"),
    )
    guidance_scale: float = Field(
        default=settings.DEFAULT_CFG,
        ge=1.0,
        le=settings.MAX_CFG,
        validation_alias=AliasChoices("guidance_scale", "cfg_scale"),
    )
    seed: Optional[int] = Field(default=None, ge=-1, le=2**32 - 1)
    num_images: int = Field(default=1, ge=1, le=settings.MAX_BATCH_SIZE)

    save_images: bool = True
    return_base64: bool = False

    @field_validator("width", "height", mode="before")
    @classmethod
    def normalize_dimensions(cls, v, info):
        if v is None or v == 0 or (isinstance(v, str) and not v.strip()):
            return settings.DEFAULT_WIDTH if info.field_name == "width" else settings.DEFAULT_HEIGHT
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            if not m:
                raise ValueError("width/height must be an integer")
            v = int(m.group())
        if v % 8 != 0:
            v = ((v + 7) // 8) * 8
        return v

    @field_validator("seed", mode="before")
    @classmethod
    def normalize_seed(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if str(v) == "-1":
            return None
        return v


class Img2ImgRequest(BaseModel):
    """Image-to-image generation request"""

    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    user_id: Optional[str] = Field(default=None, description="User ID (optional)")

    init_image: Optional[str] = Field(
        default=None,
        description="Base64 encoded source image (legacy)",
        validation_alias=AliasChoices("init_image", "image"),
    )
    init_asset_id: Optional[str] = Field(
        default=None, description="Asset ID for source image (preferred)"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path to source image (restricted to ASSETS_PATH/OUTPUT_PATH)",
    )
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    model_id: Optional[str] = Field(default=None)

    # Generation parameters (inherit defaults from config)
    width: Optional[int] = Field(default=None, ge=256, le=2048)
    height: Optional[int] = Field(default=None, ge=256, le=2048)
    num_inference_steps: int = Field(
        default=settings.DEFAULT_STEPS,
        ge=1,
        le=settings.MAX_STEPS,
        validation_alias=AliasChoices("num_inference_steps", "steps"),
    )
    guidance_scale: float = Field(
        default=settings.DEFAULT_CFG,
        ge=1.0,
        le=settings.MAX_CFG,
        validation_alias=AliasChoices("guidance_scale", "cfg_scale"),
    )
    seed: Optional[int] = Field(default=None, ge=-1, le=2**32 - 1)

    # Optional ControlNet
    controlnet: Optional[ControlNetConfig] = Field(default=None)

    @field_validator("init_image")
    @classmethod
    def validate_init_image(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return ControlNetConfig.validate_base64_image(v)

    @field_validator("init_asset_id")
    @classmethod
    def validate_init_asset_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        try:
            uuid.UUID(str(v))
        except Exception as e:
            raise ValueError(f"Invalid init_asset_id (expected UUID): {e}")
        return str(v)

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        from app.schemas.queue_requests import _resolve_allowed_image_path

        return _resolve_allowed_image_path(v)

    @model_validator(mode="after")
    def validate_init_source(self) -> "Img2ImgRequest":
        provided = [
            self.init_image is not None,
            self.init_asset_id is not None,
            self.image_path is not None,
        ]
        if sum(provided) != 1:
            raise ValueError(
                "img2img requires exactly one of: init_image (base64), init_asset_id, image_path"
            )
        return self

    @field_validator("width", "height", mode="before")
    @classmethod
    def normalize_optional_dimensions(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            if not m:
                raise ValueError("width/height must be an integer")
            v = int(m.group())
        if v % 8 != 0:
            v = ((v + 7) // 8) * 8
        return v

    @field_validator("seed", mode="before")
    @classmethod
    def normalize_seed(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if str(v) == "-1":
            return None
        return v


class CaptionRequest(BaseModel):
    """Image captioning request"""

    image: Optional[str] = None  # base64 encoded image
    image_url: Optional[str] = None
    max_length: int = Field(default=50, ge=10, le=200)

    @model_validator(mode="after")
    def validate_image_source(self):
        if not self.image and not self.image_url:
            raise ValueError("Either image or image_url must be provided")
        return self


class VQARequest(BaseModel):
    """Visual question answering request"""

    image: Optional[str] = None  # base64 encoded image
    image_url: Optional[str] = None
    question: str = Field(..., min_length=1, max_length=1000)
    max_length: int = Field(default=100, ge=10, le=500)

    @model_validator(mode="after")
    def validate_image_source(self):
        if not self.image and not self.image_url:
            raise ValueError("Either image or image_url must be provided")
        return self


class QueueSubmitRequest(BaseModel):
    """Queue task submission request"""

    task_type: str
    parameters: Dict[str, Any]
    priority: int = Field(default=0, ge=0, le=10)


class InpaintRequest(BaseModel):
    """Inpainting generation request"""

    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    user_id: Optional[str] = Field(default=None, description="User ID (optional)")

    # Required images
    init_image: Optional[str] = Field(
        default=None, description="Base64 encoded source image (legacy)"
    )
    mask_image: Optional[str] = Field(
        default=None,
        description="Base64 encoded mask (white=inpaint, black=keep) (legacy)",
    )
    init_asset_id: Optional[str] = Field(
        default=None, description="Asset ID for init image (preferred)"
    )
    mask_asset_id: Optional[str] = Field(
        default=None, description="Asset ID for mask image (preferred)"
    )
    image_path: Optional[str] = Field(
        default=None, description="Path to init image (restricted to ASSETS_PATH/OUTPUT_PATH)"
    )
    mask_path: Optional[str] = Field(
        default=None, description="Path to mask image (restricted to ASSETS_PATH/OUTPUT_PATH)"
    )

    # Inpainting specific
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    mask_blur: int = Field(default=4, ge=0, le=20, description="Mask edge blur radius")
    inpainting_fill: Literal["original", "latent_noise", "latent_nothing", "white"] = (
        Field(default="original", description="Masked area fill method")
    )

    # Standard parameters
    model_id: Optional[str] = Field(default=None)
    width: Optional[int] = Field(default=None, ge=256, le=2048)
    height: Optional[int] = Field(default=None, ge=256, le=2048)
    num_inference_steps: int = Field(default=25, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(default=None)

    @field_validator("init_image", "mask_image")
    @classmethod
    def validate_base64_images(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return ControlNetConfig.validate_base64_image(v)

    @field_validator("init_asset_id", "mask_asset_id")
    @classmethod
    def validate_asset_ids(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        try:
            uuid.UUID(str(v))
        except Exception as e:
            raise ValueError(f"Invalid asset_id (expected UUID): {e}")
        return str(v)

    @field_validator("image_path", "mask_path")
    @classmethod
    def validate_paths(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        from app.schemas.queue_requests import _resolve_allowed_image_path

        return _resolve_allowed_image_path(v)

    @model_validator(mode="after")
    def validate_sources(self) -> "InpaintRequest":
        init_sources = [
            self.init_image is not None,
            self.init_asset_id is not None,
            self.image_path is not None,
        ]
        if sum(init_sources) != 1:
            raise ValueError(
                "inpaint requires exactly one of: init_image (base64), init_asset_id, image_path"
            )

        mask_sources = [
            self.mask_image is not None,
            self.mask_asset_id is not None,
            self.mask_path is not None,
        ]
        if sum(mask_sources) != 1:
            raise ValueError(
                "inpaint requires exactly one of: mask_image (base64), mask_asset_id, mask_path"
            )
        return self


class AssetUploadRequest(BaseModel):
    """Asset upload and management request"""

    assets: List[Dict[str, Any]] = Field(..., description="List of asset uploads")
    category: Literal["reference", "mask", "pose", "depth", "custom"] = Field(
        default="reference", description="Asset category"
    )
    tags: List[str] = Field(
        default_factory=list, description="Asset tags for organization"
    )

    class AssetItem(BaseModel):
        name: str = Field(..., max_length=255)
        data: str = Field(..., description="Base64 encoded asset data")
        description: Optional[str] = Field(default="", max_length=500)

        @field_validator("data")
        @classmethod
        def validate_asset_data(cls, v: str) -> str:
            return ControlNetConfig.validate_base64_image(v)


# Response schemas remain similar but with additional metadata fields
class GenerationResponse(BaseModel):
    """Enhanced generation response with asset tracking"""

    success: bool = True
    message: str = "Generation completed successfully"
    data: Dict[str, Any] = Field(default_factory=dict)

    class GenerationData(BaseModel):
        task_id: str
        images: List[str] = Field(description="Generated image URLs/paths")
        metadata: Dict[str, Any] = Field(default_factory=dict)

        # Phase 4 additions
        controlnet_info: Optional[Dict[str, Any]] = Field(default=None)
        processing_time: Dict[str, float] = Field(
            default_factory=dict
        )  # preprocessing, generation, postprocessing
        assets_used: List[str] = Field(default_factory=list)


class BatchTxt2ImgRequest(BaseModel):
    """Request schema for batch text-to-image generation (future Phase 8)."""

    requests: List[Txt2ImgRequest] = Field(
        ..., min_length=1, max_length=10, description="List of generation requests"
    )

    # Batch control parameters
    parallel: bool = Field(
        default=False, description="Whether to process requests in parallel"
    )
    stop_on_error: bool = Field(
        default=True, description="Whether to stop on the first error"
    )


class HealthCheckRequest(BaseModel):
    """Health Check Request Schema"""

    check_gpu: bool = Field(
        default=True, description="Whether to check GPU availability"
    )
    check_models: bool = Field(
        default=False, description="Whether to check model availability"
    )
    verbose: bool = Field(
        default=False, description="Whether to return detailed status information"
    )


class ModelSwitchRequest(BaseModel):
    """Request schema for model switching."""

    model_id: str = Field(..., description="Target model ID to switch to")
    force_reload: bool = Field(
        default=False, description="Force reload the model even if already loaded"
    )


# Default negative prompts for common issues
DEFAULT_NEGATIVE_PROMPTS = {
    "quality": "blurry, low quality, worst quality, low resolution, pixelated, jpeg artifacts",
    "anatomy": "bad anatomy, deformed, mutated, extra limbs, missing limbs",
    "general": "blurry, low quality, worst quality, bad anatomy, deformed, mutated",
}

# Default style prompts for different styles
DEFAULT_STYLE_PROMPTS = {
    "photorealistic": "photorealistic, detailed, high quality, professional photography",
    "anime": "anime style, manga style, cel shading, vibrant colors",
    "artistic": "artistic, digital art, concept art, detailed illustration",
    "cinematic": "cinematic lighting, dramatic, film grain, professional",
}


class PromptTemplate(BaseModel):
    """Prompt Template Schema for reusable prompt structures"""

    name: str = Field(..., description="Template name")
    style: str = Field(
        ..., description="Template style (e.g., 'photorealistic', 'anime')"
    )
    positive_prefix: str = Field(
        default="", description="Positive prefix for the prompt"
    )
    positive_suffix: str = Field(
        default="", description="Negative suffix for the prompt"
    )
    negative_prompt: str = Field(
        default="", description="Default negative prompt to apply"
    )

    def apply_to_prompt(self, user_prompt: str) -> tuple[str, str]:
        """Apply the template to a user-provided prompt."""
        # Construct the positive prompt
        positive = (
            f"{self.positive_prefix} {user_prompt} {self.positive_suffix}".strip()
        )

        # Normalize whitespace
        positive = re.sub(r"\s+", " ", positive)
        negative = re.sub(r"\s+", " ", self.negative_prompt.strip())

        return positive, negative


# Default prompt templates for common styles
BUILTIN_TEMPLATES = [
    PromptTemplate(
        name="photorealistic",
        style="photo",
        positive_prefix="photorealistic, detailed, high quality,",
        positive_suffix=", professional photography, 8k uhd",
        negative_prompt=DEFAULT_NEGATIVE_PROMPTS["general"],
    ),
    PromptTemplate(
        name="anime",
        style="anime",
        positive_prefix="anime style, manga style,",
        positive_suffix=", vibrant colors, cel shading",
        negative_prompt=DEFAULT_NEGATIVE_PROMPTS["general"],
    ),
    PromptTemplate(
        name="artistic",
        style="art",
        positive_prefix="digital art, concept art,",
        positive_suffix=", detailed illustration, artstation",
        negative_prompt=DEFAULT_NEGATIVE_PROMPTS["general"],
    ),
]
