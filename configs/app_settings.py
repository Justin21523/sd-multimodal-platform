"""
Application configuration settings
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Multi-Modal Lab"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Shared Cache Settings
    AI_CACHE_ROOT: str = "/mnt/c/web-projects/AI_LLM_projects/ai_warehouse"

    # Model Settings
    SD_MODEL: str = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_MODEL: str = "lllyasviel/sd-controlnet-canny"
    LORA_MODELS: List[str] = []
    CAPTION_MODEL: str = "blip2"  # blip2, git, etc.
    VQA_MODEL: str = "llava"  # llava, qwen-vl, etc.
    CHAT_MODEL: str = "qwen"  # qwen, llama, etc.
    EMBEDDING_MODEL: str = "bge-m3"

    # Security Limits
    MAX_STEPS: int = 100
    MAX_RESOLUTION: int = 1024 * 1024  # 1MP
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_VIDEO_DURATION: int = 300  # 5 minutes

    # Generation Defaults
    DEFAULT_STEPS: int = 20
    DEFAULT_CFG_SCALE: float = 7.0
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512
    DEFAULT_SAMPLER: str = "DPM++ 2M Karras"

    # API Settings
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # Database
    DATABASE_URL: str = "sqlite:///./data/app.db"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
