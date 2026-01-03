"""
Queue-specific request schemas.

These schemas are used for async queue submission where large binary payloads
should be avoided. For image inputs we allow:
- base64 (legacy)
- asset_id (preferred)
- image_path (restricted to ASSETS_PATH / OUTPUT_PATH)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from app.config import settings
from app.schemas.requests import ControlNetConfig


def _resolve_allowed_image_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        raise ValueError("image_path must be an absolute path")
    if not path.exists() or not path.is_file():
        raise ValueError("image_path must exist and be a file")

    resolved = path.resolve()
    allowed_roots = [
        Path(settings.ASSETS_PATH).expanduser().resolve(),
        Path(str(settings.OUTPUT_PATH)).expanduser().resolve(),
    ]
    for root in allowed_roots:
        try:
            if resolved.is_relative_to(root):
                return str(resolved)
        except AttributeError:  # pragma: no cover (py<3.9)
            if str(resolved).startswith(str(root) + "/"):
                return str(resolved)

    raise ValueError(
        "image_path must be under ASSETS_PATH or OUTPUT_PATH (security restriction)"
    )


def _validate_uuid(value: str) -> str:
    try:
        uuid.UUID(str(value))
    except Exception as e:
        raise ValueError(f"Invalid asset_id (expected UUID): {e}")
    return str(value)


class QueueControlNetConfig(BaseModel):
    """ControlNet config for queue submissions (supports asset_id/image_path)."""

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
    def validate_image_base64(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return ControlNetConfig.validate_base64_image(v)

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _validate_uuid(v)

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _resolve_allowed_image_path(v)

    @model_validator(mode="after")
    def validate_image_source(self) -> "QueueControlNetConfig":
        provided = [
            self.image is not None,
            self.asset_id is not None,
            self.image_path is not None,
        ]
        if sum(provided) > 1:
            raise ValueError(
                "ControlNet allows at most one of: image (base64), asset_id, image_path"
            )
        return self


class QueueImg2ImgRequest(BaseModel):
    """Queue img2img request (allows init_asset_id/image_path for inputs)."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)

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
    num_images: int = Field(default=1, ge=1, le=settings.MAX_BATCH_SIZE)

    controlnet: Optional[QueueControlNetConfig] = Field(default=None)
    control_asset_id: Optional[str] = Field(
        default=None, description="Asset ID for ControlNet image (preferred)"
    )
    control_image_path: Optional[str] = Field(
        default=None,
        description="Path for ControlNet image (restricted to ASSETS_PATH/OUTPUT_PATH)",
    )

    @field_validator("init_image")
    @classmethod
    def validate_init_image_base64(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return ControlNetConfig.validate_base64_image(v)

    @field_validator("init_asset_id", "control_asset_id")
    @classmethod
    def validate_asset_ids(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _validate_uuid(v)

    @field_validator("image_path", "control_image_path")
    @classmethod
    def validate_paths(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _resolve_allowed_image_path(v)

    @model_validator(mode="after")
    def validate_sources(self) -> "QueueImg2ImgRequest":
        init_sources = [self.init_image is not None, self.init_asset_id is not None, self.image_path is not None]
        if sum(init_sources) != 1:
            raise ValueError(
                "img2img requires exactly one of: init_image (base64), init_asset_id, image_path"
            )

        if self.control_asset_id is not None or self.control_image_path is not None:
            if self.controlnet is None:
                raise ValueError(
                    "control_asset_id/control_image_path require controlnet config (type, strength, etc.)"
                )

        if self.controlnet is not None:
            cn_sources = [
                self.controlnet.image is not None,
                self.controlnet.asset_id is not None,
                self.controlnet.image_path is not None,
                self.control_asset_id is not None,
                self.control_image_path is not None,
            ]
            if sum(cn_sources) != 1:
                raise ValueError(
                    "ControlNet requires exactly one of: controlnet.image (base64), "
                    "controlnet.asset_id, controlnet.image_path, control_asset_id, control_image_path"
                )

            if (
                self.controlnet.image is None
                and self.controlnet.asset_id is None
                and self.controlnet.image_path is None
            ):
                if self.control_asset_id is not None:
                    self.controlnet.asset_id = self.control_asset_id
                    self.control_asset_id = None
                elif self.control_image_path is not None:
                    self.controlnet.image_path = self.control_image_path
                    self.control_image_path = None

        return self


class QueueInpaintRequest(BaseModel):
    """Queue inpaint request (allows asset_id/image_path for init+mask)."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)

    init_image: Optional[str] = Field(default=None, description="Base64 init image (legacy)")
    mask_image: Optional[str] = Field(
        default=None, description="Base64 mask image (legacy)"
    )
    init_asset_id: Optional[str] = Field(default=None, description="Asset ID for init image (preferred)")
    mask_asset_id: Optional[str] = Field(default=None, description="Asset ID for mask image (preferred)")
    image_path: Optional[str] = Field(
        default=None, description="Path to init image (restricted to ASSETS_PATH/OUTPUT_PATH)"
    )
    mask_path: Optional[str] = Field(
        default=None, description="Path to mask image (restricted to ASSETS_PATH/OUTPUT_PATH)"
    )

    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    mask_blur: int = Field(default=4, ge=0, le=20)
    inpainting_fill: Literal["original", "latent_noise", "latent_nothing", "white"] = Field(
        default="original"
    )

    model_id: Optional[str] = Field(default=None)
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
    num_images: int = Field(default=1, ge=1, le=settings.MAX_BATCH_SIZE)

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
        return _validate_uuid(v)

    @field_validator("image_path", "mask_path")
    @classmethod
    def validate_paths(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _resolve_allowed_image_path(v)

    @model_validator(mode="after")
    def validate_sources(self) -> "QueueInpaintRequest":
        init_sources = [self.init_image is not None, self.init_asset_id is not None, self.image_path is not None]
        if sum(init_sources) != 1:
            raise ValueError(
                "inpaint requires exactly one of: init_image (base64), init_asset_id, image_path"
            )

        mask_sources = [self.mask_image is not None, self.mask_asset_id is not None, self.mask_path is not None]
        if sum(mask_sources) != 1:
            raise ValueError(
                "inpaint requires exactly one of: mask_image (base64), mask_asset_id, mask_path"
            )
        return self


class QueueUpscaleRequest(BaseModel):
    """Queue upscale request (allows image_asset_id/image_path for inputs)."""

    image: Optional[str] = Field(default=None, description="Base64 image (legacy)")
    image_asset_id: Optional[str] = Field(
        default=None, description="Asset ID for input image (preferred)"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path to input image (restricted to ASSETS_PATH/OUTPUT_PATH)",
    )

    scale: int = Field(default=4, ge=1, le=8)
    model: str = Field(default="RealESRGAN_x4plus")
    tile_size: Optional[int] = Field(default=None, ge=0, le=2048)

    @field_validator("image")
    @classmethod
    def validate_image_base64(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return ControlNetConfig.validate_base64_image(v)

    @field_validator("image_asset_id")
    @classmethod
    def validate_image_asset_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _validate_uuid(v)

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _resolve_allowed_image_path(v)

    @model_validator(mode="after")
    def validate_image_source(self) -> "QueueUpscaleRequest":
        provided = [
            self.image is not None,
            self.image_asset_id is not None,
            self.image_path is not None,
        ]
        if sum(provided) != 1:
            raise ValueError(
                "upscale requires exactly one of: image (base64), image_asset_id, image_path"
            )
        return self


class QueueFaceRestoreRequest(BaseModel):
    """Queue face_restore request (allows image_asset_id/image_path for inputs)."""

    image: Optional[str] = Field(default=None, description="Base64 image (legacy)")
    image_asset_id: Optional[str] = Field(
        default=None, description="Asset ID for input image (preferred)"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path to input image (restricted to ASSETS_PATH/OUTPUT_PATH)",
    )

    model: str = Field(default="GFPGAN_v1.4")
    upscale: int = Field(default=2, ge=1, le=8)
    only_center_face: bool = Field(default=False)
    has_aligned: bool = Field(default=False)
    paste_back: bool = Field(default=True)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("image")
    @classmethod
    def validate_image_base64(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return ControlNetConfig.validate_base64_image(v)

    @field_validator("image_asset_id")
    @classmethod
    def validate_image_asset_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _validate_uuid(v)

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _resolve_allowed_image_path(v)

    @model_validator(mode="after")
    def validate_image_source(self) -> "QueueFaceRestoreRequest":
        provided = [
            self.image is not None,
            self.image_asset_id is not None,
            self.image_path is not None,
        ]
        if sum(provided) != 1:
            raise ValueError(
                "face_restore requires exactly one of: image (base64), image_asset_id, image_path"
            )
        return self
