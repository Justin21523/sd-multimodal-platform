# backend/config/settings.py
"""
Application Configuration Management

Handles environment variables and application settings with validation.
Uses Pydantic for type safety and automatic validation.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import torch
import os
from pathlib import Path


class Settings(BaseSettings):
    """Main application settings with environment variable support"""

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=True, description="Enable debug mode")
    reload: bool = Field(default=True, description="Enable auto-reload")

    # Model Paths
    sd_model_path: str = Field(
        default="models/stable-diffusion/sd-1.5",
        description="Stable Diffusion model path",
    )
    controlnet_model_path: str = Field(
        default="models/controlnet", description="ControlNet model path"
    )
    lora_model_path: str = Field(default="models/lora", description="LoRA model path")
    vae_model_path: str = Field(default="models/vae", description="VAE model path")

    # Hardware Settings
    device: str = Field(default="cuda", description="Compute device (cuda/cpu)")
    cuda_visible_devices: str = Field(default="0", description="CUDA device selection")
    enable_cpu_offload: bool = Field(
        default=True, description="Enable CPU offload for memory optimization"
    )
    enable_attention_slicing: bool = Field(
        default=True, description="Enable attention slicing to reduce VRAM"
    )

    # API Configuration
    api_prefix: str = Field(default="/api/v1", description="API route prefix")
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8080",
            "http://localhost:7860",
        ],
        description="CORS allowed origins",
    )
    max_workers: int = Field(default=4, description="Maximum concurrent workers")

    # Image Generation Defaults
    default_width: int = Field(default=512, description="Default image width")
    default_height: int = Field(default=512, description="Default image height")
    default_steps: int = Field(default=20, description="Default inference steps")
    default_guidance_scale: float = Field(
        default=7.5, description="Default guidance scale"
    )
    max_batch_size: int = Field(default=4, description="Maximum batch size")

    # Storage Paths
    output_dir: str = Field(
        default="data/images/output", description="Generated images output directory"
    )
    temp_dir: str = Field(
        default="data/images/temp", description="Temporary files directory"
    )
    log_dir: str = Field(default="data/logs", description="Application logs directory")

    # Security
    secret_key: str = Field(
        default="your-secret-key-here", description="JWT secret key"
    )
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration time"
    )

    @validator("device")
    def validate_device(cls, v):
        """Validate device setting and check CUDA availability"""
        if v == "cuda":
            try:
                if not torch.cuda.is_available():
                    print("⚠️ CUDA not available, falling back to CPU")
                    return "cpu"
            except ImportError:
                print("⚠️ PyTorch not installed, using CPU")
                return "cpu"
        return v

    @validator("allowed_origins")
    def validate_origins(cls, v):
        """Ensure origins are properly formatted"""
        return [origin.rstrip("/") for origin in v]

    def create_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.output_dir,
            self.temp_dir,
            self.log_dir,
            Path(self.sd_model_path).parent,
            self.controlnet_model_path,
            self.lora_model_path,
            self.vae_model_path,
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
