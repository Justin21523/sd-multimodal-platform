# app/schemas/requests.py
"""
Request schemas for SD Multi-Modal Platform API endpoints.
Defines Pydantic models for request validation and documentation.
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

    # Core parameters
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text prompt describing the desired image",
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Negative prompt to avoid unwanted elements",
    )

    # Generation parameters
    generation_params: Optional[GenerationParams] = Field(
        default_factory=GenerationParams
    )
    # Model selection
    model_id: Optional[str] = Field(
        default=None, description="Specific model to use (auto-select if None)"
    )
    #  Image generation control
    batch_size: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Batch size for generation (Phase 1: single image only)",
    )
    save_images: bool = Field(
        default=True, description="Whether to save images to disk"
    )
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
