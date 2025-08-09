# backend/schemas/requests.py
"""
API Request Models

Defines Pydantic models for request validation with:
- Type safety and automatic validation
- Clear documentation for API endpoints
- Default values and constraints
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
from backend.config.settings import Settings


class Text2ImageRequest(BaseModel):
    """Request model for text-to-image generation"""

    prompt: str = Field(
        ...,
        description="Text description of the desired image",
        min_length=1,
        max_length=1000,
        example="A beautiful sunset over mountains, digital art",
    )

    negative_prompt: Optional[str] = Field(
        default="",
        description="What to avoid in the image",
        max_length=500,
        example="blurry, low quality, distorted",
    )

    width: Optional[int] = Field(
        default=None, description="Image width in pixels", ge=64, le=2048, example=512
    )

    height: Optional[int] = Field(
        default=None, description="Image height in pixels", ge=64, le=2048, example=512
    )

    num_inference_steps: Optional[int] = Field(
        default=None,
        description="Number of denoising steps (more = higher quality, slower)",
        ge=1,
        le=100,
        example=20,
    )

    guidance_scale: Optional[float] = Field(
        default=None,
        description="How closely to follow the prompt (1-20)",
        ge=1.0,
        le=20.0,
        example=7.5,
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible results",
        ge=0,
        le=2**32 - 1,
        example=42,
    )

    batch_size: Optional[int] = Field(
        default=1,
        description="Number of images to generate",
        ge=1,
        le=Settings.max_batch_size,
        example=1,
    )

    scheduler: Optional[str] = Field(
        default="dpm", description="Diffusion scheduler algorithm", example="dpm"
    )

    @validator("width", "height")
    def validate_dimensions(cls, v):
        """Ensure dimensions are multiples of 8 for optimal performance"""
        if v is not None and v % 8 != 0:
            # Round to nearest multiple of 8
            v = ((v + 7) // 8) * 8
        return v

    @validator("scheduler")
    def validate_scheduler(cls, v):
        """Validate scheduler choice"""
        allowed_schedulers = ["dpm", "euler_a", "ddim"]
        if v not in allowed_schedulers:
            raise ValueError(f"Scheduler must be one of: {allowed_schedulers}")
        return v


class ModelSwitchRequest(BaseModel):
    """Request to switch SD model"""

    model_id: str = Field(..., description="Model identifier", example="sd-1.5")

    force_reload: bool = Field(
        default=False, description="Force reload even if model is already loaded"
    )


class SchedulerChangeRequest(BaseModel):
    """Request to change diffusion scheduler"""

    scheduler: str = Field(..., description="Scheduler algorithm name", example="dpm")

    @validator("scheduler")
    def validate_scheduler(cls, v):
        allowed_schedulers = ["dpm", "euler_a", "ddim"]
        if v not in allowed_schedulers:
            raise ValueError(f"Scheduler must be one of: {allowed_schedulers}")
        return v
