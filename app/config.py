# app/config.py
# SD Multi-Modal Platform Configuration Management
"""
SD Multi-Modal Platform Configuration Management
Use Pydantic Settings for type-safe configuration management
"""

import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import field_validator
import torch


class Settings(BaseSettings):
    """SD Multi-Modal Platform Configuration Management"""

    # API basic settings
    API_PREFIX: str = "/api/v1"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:7860",
    ]

    # Hardware settings
    DEVICE: str = "cuda"
    CUDA_VISIBLE_DEVICES: str = "0"
    ENABLE_CPU_OFFLOAD: bool = False
    USE_ATTENTION_SLICING: bool = True
    ENABLE_XFORMERS: bool = True
    TORCH_DTYPE: str = "float16"

    # MODEL PATHS
    SDXL_MODEL_PATH: str = "./models/stable-diffusion/sdxl"
    CONTROLNET_PATH: str = "./models/controlnet"
    LORA_PATH: str = "./models/lora"
    VAE_PATH: str = "./models/vae"
    UPSCALE_MODEL_PATH: str = "./models/upscale"

    # Phase 1 model selection
    PRIMARY_MODEL: Literal["sd-1.5", "sdxl-base"] = "sdxl-base"
    MODEL_CACHE_DIR: str = "./models/cache"

    # Image generation defaults
    DEFAULT_WIDTH: int = 1024
    DEFAULT_HEIGHT: int = 1024
    DEFAULT_STEPS: int = 25
    DEFAULT_CFG: float = 7.5
    DEFAULT_SAMPLER: str = "DPM++ 2M Karras"
    DEFAULT_SCHEDULER: str = "EulerDiscreteScheduler"

    # Efficency and performance settings
    MAX_WORKERS: int = 1
    MAX_BATCH_SIZE: int = 1
    MAX_QUEUE_SIZE: int = 10
    REQUEST_TIMEOUT: int = 300
    MAX_FILE_SIZE: int = 10485760  # 10MB

    # Security settings
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    JWT_EXPIRE_HOURS: int = 24
    ENABLE_NSFW_FILTER: bool = False
    ENABLE_RATE_LIMITING: bool = False

    # Storage and output settings
    OUTPUT_PATH: str = "./outputs"
    ENABLE_METADATA_LOGGING: bool = True
    KEEP_GENERATIONS_DAYS: int = 7
    SAVE_ORIGINAL_IMAGE: bool = True

    # Logging and monitoring
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    ENABLE_REQUEST_LOGGING: bool = True
    ENABLE_PROMETHEUS: bool = False

    # Development settings
    DEBUG_MODE: bool = True
    RELOAD_ON_CHANGE: bool = True

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def validate_origins(cls, v):
        """Deal with comma-separated strings for allowed origins"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("DEVICE", mode="before")
    @classmethod
    def validate_device(cls, v):
        """Validate the device setting"""
        if v == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            return "cpu"
        return v

    @field_validator("TORCH_DTYPE", mode="before")
    @classmethod
    def validate_torch_dtype(cls, v):
        """Validate the PyTorch data type setting"""
        valid_dtypes = ["float16", "float32", "bfloat16"]
        if v not in valid_dtypes:
            raise ValueError(f"TORCH_DTYPE must be one of {valid_dtypes}")
        return v

    def get_torch_dtype(self):
        """Get the PyTorch data type based on the TORCH_DTYPE setting"""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map[self.TORCH_DTYPE]

    def get_model_path(self, model_type: str = "") -> str:
        """Get the model path based on the primary model setting or specified type"""
        if model_type:
            # Check for specific model types
            if model_type == "sdxl":
                return self.SDXL_MODEL_PATH
            # Other model types can be added here as needed

        # Default to primary model path
        if self.PRIMARY_MODEL == "sdxl-base":
            return self.SDXL_MODEL_PATH
        elif self.PRIMARY_MODEL == "sd-1.5":
            return "./models/stable-diffusion/sd-1.5"

        return self.SDXL_MODEL_PATH  # Default fallback

    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            self.OUTPUT_PATH,
            f"{self.OUTPUT_PATH}/txt2img",
            f"{self.OUTPUT_PATH}/metadata",
            Path(self.LOG_FILE).parent,
            self.MODEL_CACHE_DIR,
            self.get_model_path(),
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        print(f"✅ Created directories: {len(directories)} paths")

    def get_generation_defaults(self) -> dict:
        """Get default generation parameters"""
        return {
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "num_inference_steps": self.DEFAULT_STEPS,
            "guidance_scale": self.DEFAULT_CFG,
            "scheduler": self.DEFAULT_SCHEDULER,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure directories are created at startup
settings.ensure_directories()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


# Phase 1 setup validation function
def validate_phase1_setup():
    """Validate Phase 1 setup for SD Multi-Modal Platform"""
    issues = []

    # Check device settings
    if settings.DEVICE == "cuda" and not torch.cuda.is_available():
        issues.append("CUDA not available but DEVICE=cuda")

    # Check model paths
    model_path = Path(settings.get_model_path())
    if not model_path.exists():
        issues.append(f"Model path does not exist: {model_path}")

    # Check memory settings
    if settings.DEVICE == "cuda":
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 8 and settings.PRIMARY_MODEL == "sdxl-base":
                issues.append(
                    f"GPU memory ({gpu_memory:.1f}GB) may be insufficient for SDXL"
                )
        except:
            issues.append("Cannot detect GPU memory")

    if issues:
        print("⚠️  Phase 1 Setup Issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False

    print("✅ Phase 1 configuration validated successfully")
    return True


if __name__ == "__main__":
    """Main entry point for configuration validation"""
    print("🔧 SD Multi-Modal Platform Configuration")
    print(f"Device: {settings.DEVICE}")
    print(f"Primary Model: {settings.PRIMARY_MODEL}")
    print(f"Model Path: {settings.get_model_path()}")
    print(f"Output Path: {settings.OUTPUT_PATH}")

    validate_phase1_setup()
