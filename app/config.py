# app/config.py
"""
Configuration management for SD Multi-Modal Platform Phase 3.
Handles environment variables and system settings with Pydantic v2.
"""


import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch
import torch.version


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    All settings can be overridden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    # API Configuration
    API_PREFIX: str = Field(default="/api/v1", description="API route prefix")
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000",
        description="CORS allowed origins (comma-separated)",
    )

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE_PATH: str = Field(default="logs/app.log", description="Log file path")
    ENABLE_REQUEST_LOGGING: bool = Field(
        default=True, description="Enable request logging middleware"
    )

    # Hardware Configuration - RTX 5080 (sm_120) Optimized
    DEVICE: str = Field(default="cuda", description="PyTorch device (cuda/cpu)")
    TORCH_DTYPE: str = Field(default="float16", description="PyTorch data type")
    USE_SDPA: bool = Field(
        default=True,
        description="Use PyTorch native SDPA (Scaled Dot Product Attention)",
    )
    ENABLE_XFORMERS: bool = Field(
        default=False,
        description="Enable xFormers optimization (auto-detect if available)",
    )
    USE_ATTENTION_SLICING: bool = Field(
        default=True, description="Enable attention slicing"
    )
    ENABLE_CPU_OFFLOAD: bool = Field(
        default=False, description="Enable CPU offload for models"
    )
    ENABLE_MODEL_COMPILATION: bool = Field(
        default=False, description="Enable PyTorch 2.0+ model compilation"
    )

    # Model Configuration
    PRIMARY_MODEL: Literal["sdxl-base", "sd-1.5", "sd-2.1"] = Field(
        default="sdxl-base", description="Primary model to load on startup"
    )

    # Model paths
    SD_MODEL_PATH: str = Field(default="./models/stable-diffusion")
    SDXL_MODEL_PATH: str = Field(default="./models/sdxl")
    CONTROLNET_PATH: str = Field(default="./models/controlnet")
    LORA_PATH: str = Field(default="./models/lora")
    VAE_PATH: str = Field(default="./models/vae")
    UPSCALE_MODEL_PATH: str = Field(
        default="./models/upscale", description="Upscale models directory"
    )

    # Generation Defaults
    DEFAULT_WIDTH: int = Field(default=1024, description="Default image width")
    DEFAULT_HEIGHT: int = Field(default=1024, description="Default image height")
    DEFAULT_STEPS: int = Field(default=25, description="Default inference steps")
    DEFAULT_CFG: float = Field(default=7.5, description="Default CFG scale")
    DEFAULT_SAMPLER: str = Field(
        default="DPM++ 2M Karras", description="Default sampler"
    )

    # Performance Limits
    MAX_WORKERS: int = Field(default=1, description="Maximum concurrent workers")
    MAX_BATCH_SIZE: int = Field(default=4, ge=1, le=8)
    MAX_QUEUE_SIZE: int = Field(default=100, description="Maximum queue size")
    REQUEST_TIMEOUT: int = Field(default=300, description="Request timeout in seconds")
    MAX_FILE_SIZE: int = Field(
        default=10 * 1024 * 1024, description="Maximum upload file size in bytes"
    )

    # Storage Configuration
    OUTPUT_PATH: Path = Field(
        default=Path("./outputs"), description="Output files directory"
    )
    ASSETS_PATH: Path = Field(default=Path("./assets"), description="Assets directory")
    ENABLE_METADATA_LOGGING: bool = Field(
        default=True, description="Enable metadata logging"
    )
    KEEP_GENERATIONS_DAYS: int = Field(
        default=7, description="Days to keep generated files"
    )

    # Security Configuration
    SECRET_KEY: str = Field(
        default="your-super-secret-key-change-this", description="JWT secret key"
    )
    JWT_EXPIRE_HOURS: int = Field(default=24, description="JWT token expiration hours")
    ENABLE_NSFW_FILTER: bool = Field(
        default=True, description="Enable NSFW content filtering"
    )
    ENABLE_RATE_LIMITING: bool = Field(
        default=True, description="Enable API rate limiting"
    )

    # Monitoring
    ENABLE_PROMETHEUS: bool = Field(default=False)
    ENABLE_HEALTH_CHECKS: bool = Field(
        default=True, description="Enable health check endpoints"
    )
    SENTRY_DSN: str = Field(default="")

    # Development settings
    DEBUG: bool = True
    RELOAD: bool = True

    # Model Download Settings
    HF_TOKEN: str = Field(default="")
    USE_AUTH_TOKEN: bool = Field(default=False)
    CACHE_DIR: str = Field(default="./cache")

    # Phase 3 Specific Settings
    AUTO_MODEL_SWITCHING: bool = Field(default=True)
    PRELOAD_MODELS: bool = Field(default=False)
    ENABLE_MODEL_CACHING: bool = Field(default=True)
    GENERATION_QUEUE_SIZE: int = Field(default=100)

    # Experimental Features
    ENABLE_CONTROLNET: bool = Field(default=False)
    ENABLE_LORA: bool = Field(default=False)
    ENABLE_VIDEO_GENERATION: bool = Field(default=False)
    ENABLE_BATCH_API: bool = Field(default=False)

    @field_validator("DEVICE", mode="before")
    @classmethod
    def validate_device(cls, v):
        """Validate and auto-detect device if needed."""
        if v == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            return "cpu"
        return v

    @field_validator("TORCH_DTYPE", mode="before")
    @classmethod
    def validate_torch_dtype(cls, v):
        """Validate PyTorch data type."""
        valid_dtypes = ["float32", "float16", "bfloat16"]
        if v not in valid_dtypes:
            raise ValueError(f"TORCH_DTYPE must be one of {valid_dtypes}")
        return v

    @field_validator("DEFAULT_WIDTH", "DEFAULT_HEIGHT", mode="before")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Ensure dimensions are multiples of 8."""
        return ((v + 7) // 8) * 8

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @computed_field
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse CORS origins from string to list."""
        if isinstance(self.ALLOWED_ORIGINS, str):
            origins = [
                origin.strip()
                for origin in self.ALLOWED_ORIGINS.split(",")
                if origin.strip()
            ]
            return origins
        return []

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to PyTorch dtype."""
        dtype_mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_mapping[self.TORCH_DTYPE]

    def get_model_path(self, model_type: str = "primary") -> str:
        """Get model path for specified type."""
        if model_type == "primary":
            if self.PRIMARY_MODEL == "sdxl-base":
                return f"{self.SDXL_MODEL_PATH}/sdxl-base"
            elif self.PRIMARY_MODEL in ["sd-1.5", "sd-2.1", "custom"]:
                return f"{self.SD_MODEL_PATH}/{self.PRIMARY_MODEL}"
        elif model_type == "controlnet":
            return self.CONTROLNET_PATH
        elif model_type == "lora":
            return self.LORA_PATH
        elif model_type == "vae":
            return self.VAE_PATH

        return self.SD_MODEL_PATH

    def get_allowed_origins(self) -> List[str]:
        """Get CORS allowed origins as a list."""
        return self.allowed_origins_list

    def ensure_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.OUTPUT_PATH,
            f"{self.OUTPUT_PATH}/txt2img",
            f"{self.OUTPUT_PATH}/img2img",
            f"{self.OUTPUT_PATH}/inpaint",
            f"{self.OUTPUT_PATH}/upscale",
            Path(self.LOG_FILE_PATH).parent,
            self.CACHE_DIR,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_generation_defaults(self) -> dict:
        """Get default generation parameters."""
        return {
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "num_inference_steps": self.DEFAULT_STEPS,
            "guidance_scale": self.DEFAULT_CFG,
            "sampler": self.DEFAULT_SAMPLER,
        }

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.DEBUG or self.RELOAD

    def get_vram_estimate(self, model_id: str, width: int, height: int) -> float:
        """Estimate VRAM usage for given parameters."""
        # Base VRAM estimates in GB
        base_vram = {"sd-1.5": 4.0, "sd-2.1": 6.0, "sdxl-base": 8.0}

        # Resolution multiplier
        pixels = width * height
        base_pixels = 512 * 512
        resolution_multiplier = pixels / base_pixels

        # Data type multiplier
        dtype_multiplier = 0.5 if self.TORCH_DTYPE == "float16" else 1.0

        estimated_vram = (
            base_vram.get(model_id, 8.0) * resolution_multiplier * dtype_multiplier
        )

        # Add overhead for optimizations
        if self.USE_ATTENTION_SLICING:
            estimated_vram *= 0.8
        if self.ENABLE_CPU_OFFLOAD:
            estimated_vram *= 0.6

        return round(estimated_vram, 1)

    def get_device_info(self) -> dict:
        """Get device information for health checks."""
        info = {
            "device": self.DEVICE,
            "torch_dtype": self.TORCH_DTYPE,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                }
            )
            if torch.cuda.device_count() > 0:
                info.update(
                    {
                        "gpu_name": torch.cuda.get_device_name(0),
                        "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                    }
                )

        return info


# Create global settings instance
settings = Settings()
# Ensure directories are created at startup
settings.ensure_directories()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


if __name__ == "__main__":
    """Main entry point for configuration validation"""
    print("🔧 SD Multi-Modal Platform Configuration")
    print(f"Device: {settings.DEVICE}")
    print(f"Primary Model: {settings.PRIMARY_MODEL}")
    print(f"Use SDPA: {settings.USE_SDPA}")
    print(f"Enable xformers: {settings.ENABLE_XFORMERS}")
    print(f"Model Path: {settings.get_model_path()}")
    print(f"Output Path: {settings.OUTPUT_PATH}")
