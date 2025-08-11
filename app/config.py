# app/config.py
"""
Configuration management with Pydantic settings for environment variables.
Phase 2: Backend Framework & Basic API Services
Updated with logging and API framework settings.
"""

import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import field_validator, Field
import torch
import torch.version


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    All settings can be overridden via environment variables.
    """

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="logs/app.log", description="Log file path")
    ENABLE_REQUEST_LOGGING: bool = Field(
        default=True, description="Enable request logging middleware"
    )

    # API basic settings
    API_PREFIX: str = "/api/v1"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:7860",
    ]

    # Hardware Configuration
    DEVICE: str = Field(default="cuda", description="PyTorch device (cuda/cpu)")
    TORCH_DTYPE: str = Field(default="float16", description="PyTorch data type")
    ENABLE_XFORMERS: bool = Field(
        default=True, description="Enable xFormers optimization"
    )
    USE_ATTENTION_SLICING: bool = Field(
        default=True, description="Enable attention slicing"
    )
    ENABLE_CPU_OFFLOAD: bool = Field(
        default=False, description="Enable CPU offload for models"
    )
    CUDA_VISIBLE_DEVICES: str = "0"

    # Model Configuration
    PRIMARY_MODEL: Literal["sd-1.5", "sdxl-base"] = Field(
        default="sdxl-base", description="Primary model to load on startup"
    )
    SD_MODEL_PATH: Path = Field(
        default=Path("./models/stable-diffusion/sd-1.5"),
        description="SD 1.5 models directory",
    )
    SDXL_MODEL_PATH: Path = Field(
        default=Path("./models/stable-diffusion/sdxl"),
        description="SDXL models directory",
    )
    CONTROLNET_PATH: Path = Field(
        default=Path("./models/controlnet"), description="ControlNet models directory"
    )
    LORA_PATH: Path = Field(
        default=Path("./models/lora"), description="LoRA models directory"
    )
    VAE_PATH: Path = Field(
        default=Path("./models/vae"), description="VAE models directory"
    )
    UPSCALE_MODEL_PATH: Path = Field(
        default=Path("./models/upscale"), description="Upscale models directory"
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
    MAX_BATCH_SIZE: int = Field(default=1, description="Maximum batch size per request")
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

    # Feature Flags
    ENABLE_PROMETHEUS: bool = Field(
        default=False, description="Enable Prometheus metrics"
    )
    ENABLE_HEALTH_CHECKS: bool = Field(
        default=True, description="Enable health check endpoints"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

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

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def validate_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to PyTorch dtype."""
        dtype_mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_mapping[self.TORCH_DTYPE]

    def get_model_path(self, model_type: str = "") -> Path:
        """Get model path based on type or primary model."""
        if model_type is None:
            model_type = self.PRIMARY_MODEL

        path_mapping = {
            "sd-1.5": self.SD_MODEL_PATH,
            "sdxl-base": self.SDXL_MODEL_PATH,
            "controlnet": self.CONTROLNET_PATH,
            "lora": self.LORA_PATH,
            "vae": self.VAE_PATH,
            "upscale": self.UPSCALE_MODEL_PATH,
        }

        return path_mapping.get(model_type, self.SD_MODEL_PATH)

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.OUTPUT_PATH,
            self.ASSETS_PATH,
            self.SD_MODEL_PATH,
            self.SDXL_MODEL_PATH,
            self.CONTROLNET_PATH,
            self.LORA_PATH,
            self.VAE_PATH,
            self.UPSCALE_MODEL_PATH,
            Path("logs"),  # For log files
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_generation_defaults(self) -> dict:
        """Get default generation parameters as dictionary."""
        return {
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "steps": self.DEFAULT_STEPS,
            "cfg_scale": self.DEFAULT_CFG,
            "sampler": self.DEFAULT_SAMPLER,
        }

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
    print(f"Model Path: {settings.get_model_path()}")
    print(f"Output Path: {settings.OUTPUT_PATH}")
