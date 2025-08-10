# app/schemas/requests.py
"""
SD Multi-Modal Platform - API Request Schemas
This module defines the request schemas for the text-to-image generation API.
"""

from typing import Optional, List, Literal, Annotated
from pydantic import BaseModel, field_validator, Field, conlist, Field
from pydantic_settings import BaseSettings
import re
import random


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

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, v):
        """Ensure seed is a valid integer or None, default to random if None"""
        if v is None or v == -1:
            return random.randint(0, 2**32 - 1)
        return max(0, min(v, 2**32 - 1))

    @field_validator("width", "height", mode="before")
    @classmethod
    def validate_dimensions(cls, v):
        """Ensure dimensions are multiples of 8 for compatibility (e.g., Stable Diffusion requires this)"""
        return (v // 8) * 8


class Txt2ImgRequest(BaseModel):
    """Text-to-Image Generation Request Schema"""

    # Core parameters
    prompt: str = Field(
        ..., min_length=1, max_length=1000, description="Prompt for image generation"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Negative prompt to exclude from generation",
    )

    # Generation parameters
    generation_params: Optional[GenerationParams] = Field(
        default_factory=GenerationParams
    )

    # Phase 1: Model selection
    model_id: Optional[str] = Field(
        default=None, description="Assigned model ID for this request"
    )

    # Phase 1: Image generation control
    batch_size: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Batch size for generation (Phase 1: single image only)",
    )

    # Phase 1: Additional options
    save_metadata: bool = Field(
        default=True, description="Whether to save generation metadata"
    )
    return_base64: bool = Field(
        default=False, description="Return images as base64 strings instead of URLs"
    )

    @field_validator("prompt", mode="before")
    @classmethod
    def validate_prompt(cls, v):
        """Prompt validation and basic filtering"""
        # Basic whitespace normalization
        v = re.sub(r"\s+", " ", v.strip())

        # Basic content filtering (e.g., no NSFW content)
        forbidden_patterns = [
            r"\b(nsfw|nude|sex)\b",  # Basic NSFW filter
        ]

        for pattern in forbidden_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Prompt contains forbidden content")

        return v

    @field_validator("negative_prompt", mode="before")
    @classmethod
    def validate_negative_prompt(cls, v):
        """Negative prompt validation and basic filtering"""
        if v is None:
            return ""
        return re.sub(r"\s+", " ", v.strip())

    def get_effective_params(self) -> dict:
        """Get effective parameters for the request, excluding None values"""
        if self.generation_params is not None:
            params = self.generation_params.model_dump()
        else:
            params = {}

        # Include model_id if specified
        return {k: v for k, v in params.items() if v is not None}

    def get_prompt_hash(self) -> str:
        """Generate a unique hash for the prompt and parameters for caching purposes"""
        import hashlib

        content = f"{self.prompt}|{self.negative_prompt}|{self.generation_params.model_dump() if self.generation_params else ''}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class BatchTxt2ImgRequest(BaseModel):
    """Bathch Text-to-Image Generation Request Schema (Phase 1)"""

    requests: List[Txt2ImgRequest] = Field(
        ..., description="List of text-to-image generation requests"
    )  # Phase 1 limits to 1 request

    # Batch control parameters
    parallel_execution: bool = Field(
        default=False, description="Whether to run requests in parallel"
    )
    stop_on_error: bool = Field(
        default=True, description="Whether to stop on the first error"
    )

    @field_validator("requests", mode="before")
    @classmethod
    def validate_batch_size(cls, v):
        """Phase 1 limits batch size to 1 request only"""
        if len(v) > 1:
            raise ValueError("Phase 1 only supports single request batches")
        return v


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


# Phase 1 Supported Models
PHASE1_SUPPORTED_MODELS = [
    "sdxl-base",
    "sd-1.5",  # Optional: can be added later
]


class ModelSwitchRequest(BaseModel):
    """Model Switch Request Schema (Phase 1)"""

    model_id: str = Field(..., description="Model ID to switch to")
    force_reload: bool = Field(
        default=False, description="Force reload the model even if already loaded"
    )

    @field_validator("model_id", mode="before")
    @classmethod
    def validate_model_id(cls, v):
        """Validate the model ID against supported models"""
        if v not in PHASE1_SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {v}. Supported: {PHASE1_SUPPORTED_MODELS}"
            )
        return v


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
