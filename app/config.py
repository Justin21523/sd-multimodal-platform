# app/config.py
"""
Configuration management with Phase 5 queue and post-processing settings
"""


import os
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any
from pydantic import Field, field_validator, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import re
import torch
import torch.version


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    All settings can be overridden via environment variables.
    """

    APP_NAME: str = "SD Multi-Modal Platform"
    APP_VERSION: str = "0.1.0"
    PHASE: str = "Phase 6: Queue System & Rate Limiting"
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="allow"
    )
    DEBUG: bool = False

    # Shared Cache / Storage Roots (must follow ~/Desktop/data_model_structure.md)
    # - caches: /mnt/c/ai_cache
    # - models: /mnt/c/ai_models
    # - outputs: /mnt/data/...
    AI_CACHE_ROOT: str = "/mnt/c/ai_cache"
    AI_MODELS_ROOT: str = "/mnt/c/ai_models"
    AI_OUTPUT_ROOT: str = "/mnt/data/training/runs/sd-multimodal-platform"

    # Model Settings
    SD_MODEL: str = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_MODEL: str = "lllyasviel/sd-controlnet-canny"
    CAPTION_MODEL: str = "Salesforce/blip2-opt-2.7b"
    VQA_MODEL: str = "llava-hf/llava-1.5-7b-hf"
    CHAT_MODEL: str = "Qwen/Qwen2-7B-Instruct"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # API Configuration
    API_PREFIX: str = Field(default="/api/v1", description="API route prefix")
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8080,http://localhost:5173,http://localhost:4173,http://127.0.0.1:3000,http://127.0.0.1:5173",
        description="CORS allowed origins (comma-separated)",
    )

    # Device and Precision
    DEVICE: str = "auto"  # auto, cuda, cpu
    PRECISION: str = "float16"  # float32, float16, bfloat16
    ENABLE_4BIT: bool = False
    ENABLE_8BIT: bool = False

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    # === Environment and Mode Settings ===
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development", description="Environment mode"
    )
    MINIMAL_MODE: bool = Field(
        default=False, description="Run in minimal mode without model loading"
    )
    DEBUG: bool = Field(default=True, description="Debug mode")
    RELOAD: bool = True
    TESTING: bool = False
    MOCK_GENERATION: bool = False  # Use mock responses for testing

    # Database
    DATABASE_URL: str = "sqlite:///./data/app.db"

    # Offline Mode
    OFFLINE_MODE: bool = False

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

    # Model paths (Path objects used by validators/helpers)
    MODEL_BASE_PATH: Path = Path("/mnt/c/ai_models")
    OUTPUT_BASE_PATH: Path = Path(
        "/mnt/data/training/runs/sd-multimodal-platform/outputs"
    )
    CACHE_BASE_PATH: Path = Path("/mnt/c/ai_cache")

    # Primary models
    PRIMARY_SD_MODEL: str = "sdxl-base"  # "sdxl-base", "sd-1.5", "pixart-sigma"
    FALLBACK_SD_MODEL: str = "sd-1.5"

    # Model Configuration
    PRIMARY_MODEL: Literal["sdxl-base", "sd-1.5", "sd-2.1"] = Field(
        default="sdxl-base", description="Primary model to load on startup"
    )
    PRIMARY_SD_MODEL: str = "sdxl-base"  # "sdxl-base", "sd-1.5", "pixart-sigma"
    FALLBACK_SD_MODEL: str = "sd-1.5"

    # === Path Configuration ===
    MODELS_PATH: str = Field(
        default="/mnt/c/ai_models", description="Base models directory"
    )
    SD_MODEL_PATH: str = Field(
        default="/mnt/c/ai_models/stable-diffusion", description="SD models path"
    )
    SDXL_MODEL_PATH: str = Field(
        default="/mnt/c/ai_models/stable-diffusion/sdxl",
        description="SDXL models path",
    )
    CONTROLNET_PATH: str = Field(
        default="/mnt/c/ai_models/controlnet", description="ControlNet models path"
    )
    LORA_PATH: str = Field(
        default="/mnt/c/ai_models/lora", description="LoRA models path"
    )
    VAE_PATH: str = Field(default="/mnt/c/ai_models/vae", description="VAE models path")
    OUTPUT_PATH: str = Field(
        default="/mnt/data/training/runs/sd-multimodal-platform/outputs",
        description="Generated outputs path",
    )

    SD15_MODEL_PATH: Optional[str] = None
    PIXART_MODEL_PATH: Optional[str] = None
    DEEPFLOYD_MODEL_PATH: Optional[str] = None
    STABLE_CASCADE_MODEL_PATH: Optional[str] = None

    # === Post-processing Paths ===
    UPSCALE_MODELS_PATH: str = Field(
        default="/mnt/c/ai_models/upscale", description="Upscale models directory"
    )
    FACE_RESTORE_MODELS_PATH: str = Field(
        default="/mnt/c/ai_models/face-restore",
        description="Face restoration models directory",
    )

    # Generation Defaults
    DEFAULT_WIDTH: int = Field(default=1024, description="Default image width")
    DEFAULT_CFG_SCALE: float = 7.0
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
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    MAX_REQUEST_SIZE: int = 100 * 1024 * 1024  # 100MB

    # === Queue Configuration ===
    ENABLE_QUEUE: bool = Field(default=True, description="Enable task queue system")
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    REDIS_URL: Optional[str] = None

    # Queue settings
    MAX_CONCURRENT_TASKS: int = 4  # Maximum tasks running simultaneously
    QUEUE_RESULT_TTL: int = 3600  # Result storage time in seconds (1 hour)
    QUEUE_TASK_TTL: int = 86400 * 7  # Task info storage time (7 days)

    # Rate limiting
    RATE_LIMIT_PER_HOUR: int = 100  # Max requests per user per hour
    RATE_LIMIT_BURST: int = 10  # Burst allowance
    GLOBAL_RATE_LIMIT_PER_MINUTE: int = 1000  # Global rate limit
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_MINUTES: int = 1
    MAX_CONCURRENT_TASKS: int = 4

    # Security Limits
    MAX_STEPS: int = 100
    MAX_CFG: float = 20.0
    MAX_RESOLUTION: int = 1024 * 1024  # 1MP
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_UPLOAD_MB: int = 10
    MAX_VIDEO_DURATION: int = 300  # 5 minutes

    # === Celery Configuration ===
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0", description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/0", description="Celery result backend"
    )
    # Worker settings
    CELERY_WORKER_CONCURRENCY: int = 2  # Workers per process
    CELERY_MAX_TASKS_PER_CHILD: int = 50  # Tasks before worker restart
    CELERY_MAX_MEMORY_PER_CHILD: int = 8 * 1024 * 1024  # 8GB memory limit

    # Task timeouts
    CELERY_TASK_SOFT_TIME_LIMIT: int = 1800  # 30 minutes soft limit
    CELERY_TASK_TIME_LIMIT: int = 2400  # 40 minutes hard limit

    # Queue configuration
    CELERY_GENERATION_QUEUE: str = "generation"
    CELERY_POSTPROCESS_QUEUE: str = "postprocess"
    CELERY_MAINTENANCE_QUEUE: str = "maintenance"

    # === Task Queue Settings ===
    TASK_TIME_LIMIT: int = Field(default=3600, description="Task time limit in seconds")
    TASK_SOFT_TIME_LIMIT: int = Field(
        default=3300, description="Task soft time limit in seconds"
    )
    WORKER_MAX_TASKS: int = Field(
        default=10, description="Max tasks per worker before restart"
    )
    TASK_RETENTION_HOURS: int = Field(
        default=72, description="Task retention period in hours"
    )

    # === Post-processing Settings ===
    ENABLE_POSTPROCESS: bool = Field(
        default=True, description="Enable post-processing features"
    )
    POSTPROCESS_MAX_IMAGES: int = Field(
        default=10, description="Max images per post-processing task"
    )
    UPSCALE_MAX_SCALE: int = Field(default=4, description="Maximum upscale factor")

    # === Batch Processing Settings ===
    BATCH_TIMEOUT: int = Field(
        default=7200, description="Batch processing timeout in seconds"
    )

    # === Memory Management ===
    VRAM_CLEANUP_THRESHOLD: float = Field(
        default=0.8, description="VRAM usage threshold for cleanup"
    )
    AUTO_CLEANUP_MODELS: bool = Field(
        default=True, description="Automatically unload models when needed"
    )
    ENABLE_ATTENTION_SLICING: bool = True
    ENABLE_CPU_OFFLOAD: bool = True
    ENABLE_XFORMERS: bool = True
    ENABLE_TORCH_COMPILE: bool = False  # Experimental
    ENABLE_VAE_SLICING: bool = True

    # GPU memory settings
    GPU_MEMORY_FRACTION: float = 0.9  # Use 90% of GPU memory
    GPU_MEMORY_GROWTH: bool = True  # Allow memory growth
    FORCE_GPU_CLEANUP: bool = True  # Force cleanup after each task

    # Model loading settings
    MODEL_CPU_OFFLOAD: bool = True
    MODEL_SEQUENTIAL_CPU_OFFLOAD: bool = False
    MODEL_ENABLE_MODEL_CPU_OFFLOAD: bool = True

    # =====================================
    # File Management Settings
    # =====================================

    MAX_CONCURRENT_DOWNLOADS: int = 5

    # Cleanup settings
    AUTO_CLEANUP_ENABLED: bool = True
    CLEANUP_INTERVAL_HOURS: int = 24
    KEEP_OUTPUTS_DAYS: int = 7
    KEEP_TEMP_FILES_HOURS: int = 2

    # Image format settings
    OUTPUT_FORMAT: str = "PNG"  # "PNG", "JPEG", "WEBP"
    OUTPUT_QUALITY: int = 95  # For JPEG/WEBP
    SAVE_ORIGINAL_PARAMS: bool = True
    SAVE_METADATA: bool = True

    # CORS settings
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # Security Configuration
    SECRET_KEY: str = Field(
        default="your-super-secret-key-change-this", description="JWT secret key"
    )
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    JWT_EXPIRE_HOURS: int = Field(default=24, description="JWT token expiration hours")
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    ENABLE_NSFW_FILTER: bool = Field(
        default=True, description="Enable NSFW content filtering"
    )
    ENABLE_RATE_LIMITING: bool = Field(
        default=True, description="Enable API rate limiting"
    )
    # Observability
    ENABLE_REQUEST_ID: bool = Field(
        default=True, description="Attach request IDs to logs/responses"
    )
    LOG_BODY_LIMIT_BYTES: int = Field(
        default=4096, description="Max request body bytes to log"
    )

    # === Logging and Monitoring ===
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = "json"  # "json" or "text"
    LOG_FILE_PATH: str = Field(
        default="/mnt/data/training/runs/sd-multimodal-platform/logs/app.log",
        description="Log file path",
    )
    ENABLE_REQUEST_LOGGING: bool = Field(
        default=True, description="Enable request logging"
    )
    ENABLE_PROMETHEUS: bool = Field(
        default=False, description="Enable Prometheus metrics"
    )
    SENTRY_DSN: Optional[str] = Field(
        default=None, description="Sentry DSN for error tracking"
    )

    # Metrics and monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    PROMETHEUS_MULTIPROC_DIR: Optional[str] = None

    # Health check settings
    HEALTH_CHECK_INTERVAL: int = 60  # seconds
    HEALTH_CHECK_TIMEOUT: int = 30  # seconds

    # Storage Configuration
    ASSETS_PATH: Path = Field(
        default=Path("/mnt/data/training/runs/sd-multimodal-platform/assets"),
        description="Assets directory",
    )
    ENABLE_METADATA_LOGGING: bool = Field(
        default=True, description="Enable metadata logging"
    )
    KEEP_GENERATIONS_DAYS: int = Field(
        default=7, description="Days to keep generated files"
    )

    # Model Download Settings
    HF_TOKEN: str = Field(default="")
    USE_AUTH_TOKEN: bool = Field(default=False)
    CACHE_DIR: str = Field(default="/mnt/c/ai_cache")

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

    # # Phase 4 Specific Settings
    CONTROLNET_MEMORY_EFFICIENT: bool = Field(default=True)
    MAX_CONTROLNET_PROCESSORS: int = Field(default=2, ge=1, le=4)
    ENABLE_PROGRESSIVE_LOADING: bool = Field(default=True)
    ASSET_CACHE_SIZE_MB: int = Field(default=500, ge=100, le=2000)

    @model_validator(mode="after")
    def _derive_storage_paths(self) -> "Settings":
        """Derive all storage paths from AI_*_ROOTs (single source of truth)."""
        models_root = Path(str(self.AI_MODELS_ROOT)).expanduser()
        cache_root = Path(str(self.AI_CACHE_ROOT)).expanduser()
        output_root = Path(str(self.AI_OUTPUT_ROOT)).expanduser()

        # Models and caches must live under /mnt/c per ~/Desktop/data_model_structure.md
        self.MODELS_PATH = str(models_root)
        self.CONTROLNET_PATH = str(models_root / "controlnet")
        self.UPSCALE_MODELS_PATH = str(models_root / "upscale")
        self.FACE_RESTORE_MODELS_PATH = str(models_root / "face-restore")

        # Derived model subpaths
        self.SD_MODEL_PATH = str(models_root / "stable-diffusion")
        self.SDXL_MODEL_PATH = str(models_root / "stable-diffusion" / "sdxl")
        self.LORA_PATH = str(models_root / "lora")
        self.VAE_PATH = str(models_root / "vae")

        # Outputs must live under /mnt/data
        self.OUTPUT_PATH = str(output_root / "outputs")
        self.ASSETS_PATH = Path(output_root / "assets")
        self.LOG_FILE_PATH = str(output_root / "logs" / "app.log")

        # Keep legacy path objects in sync
        self.MODEL_BASE_PATH = models_root
        self.CACHE_BASE_PATH = cache_root
        self.CACHE_DIR = str(cache_root)
        self.OUTPUT_BASE_PATH = Path(output_root / "outputs")

        return self

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
    def validate_dimensions(cls, v):
        """Ensure dimensions are multiples of 8."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None  # Default to None if empty
        if isinstance(v, str):
            # Allow ""1024" "1024px" or similar formats
            m = re.search(r"\d+", v)
            if not m:
                raise ValueError("DEFAULT_* must be an integer")
            v = int(m.group())
        return ((v + 7) // 8) * 8

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("REDIS_HOST", mode="before")
    @classmethod
    def validate_redis_host(cls, v: str) -> str:
        """Validate Redis host"""
        if not v or v.strip() == "":
            return "localhost"
        return v.strip()

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

    @computed_field
    @property
    def models_path_obj(self) -> Path:
        """Get models path as Path object"""
        return Path(self.MODELS_PATH)

    @computed_field
    @property
    def redis_url(self) -> str:
        """Construct Redis URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.REDIS_URL:
            return self.REDIS_URL

        auth_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to PyTorch dtype."""
        dtype_mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_mapping[self.TORCH_DTYPE]

    def get_model_path(self, model_type: str = "primary") -> str:
        """Get model path by type"""
        path_map = {
            "sd": self.SD_MODEL_PATH,
            "sdxl": self.SDXL_MODEL_PATH,
            "controlnet": self.CONTROLNET_PATH,
            "lora": self.LORA_PATH,
            "vae": self.VAE_PATH,
            "upscale": self.UPSCALE_MODELS_PATH,
            "face_restore": self.FACE_RESTORE_MODELS_PATH,
        }
        return path_map.get(model_type, self.SD_MODEL_PATH)

    def get_allowed_origins(self) -> List[str]:
        """Get CORS allowed origins as a list."""
        return self.allowed_origins_list

    def ensure_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.MODELS_PATH,
            self.SD_MODEL_PATH,
            self.SDXL_MODEL_PATH,
            self.CONTROLNET_PATH,
            self.LORA_PATH,
            self.VAE_PATH,
            self.UPSCALE_MODELS_PATH,
            self.FACE_RESTORE_MODELS_PATH,
            self.OUTPUT_PATH,
            f"{self.OUTPUT_PATH}/txt2img",
            f"{self.OUTPUT_PATH}/img2img",
            f"{self.OUTPUT_PATH}/inpaint",
            f"{self.OUTPUT_PATH}/upscale",
            f"{self.OUTPUT_PATH}/face_restore",
            f"{self.OUTPUT_PATH}/postprocess",
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

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT == "production"

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

    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration for models"""
        import torch

        config = {
            "device": self.DEVICE,
            "dtype": getattr(torch, self.TORCH_DTYPE),
        }

        if self.DEVICE == "cuda" and torch.cuda.is_available():
            config.update(
                {
                    "device_id": 0,
                    "memory_fraction": self.GPU_MEMORY_FRACTION,
                    "allow_growth": self.GPU_MEMORY_GROWTH,
                }
            )

        return config

    def get_model_optimization_config(self) -> Dict[str, bool]:
        """Get model optimization settings"""
        return {
            "attention_slicing": self.ENABLE_ATTENTION_SLICING,
            "cpu_offload": self.ENABLE_CPU_OFFLOAD,
            "xformers": self.ENABLE_XFORMERS,
            "torch_compile": self.ENABLE_TORCH_COMPILE,
            "sequential_cpu_offload": self.MODEL_SEQUENTIAL_CPU_OFFLOAD,
            "model_cpu_offload": self.MODEL_ENABLE_MODEL_CPU_OFFLOAD,
        }

    def get_celery_config(self) -> Dict[str, Any]:
        """Get Celery configuration"""
        redis_url = self.get_redis_url()

        return {
            "broker_url": redis_url,
            "result_backend": redis_url,
            "task_serializer": "json",
            "accept_content": ["json"],
            "result_serializer": "json",
            "timezone": "UTC",
            "enable_utc": True,
            "worker_concurrency": self.CELERY_WORKER_CONCURRENCY,
            "worker_prefetch_multiplier": 1,
            "task_acks_late": True,
            "task_soft_time_limit": self.CELERY_TASK_SOFT_TIME_LIMIT,
            "task_time_limit": self.CELERY_TASK_TIME_LIMIT,
            "worker_max_tasks_per_child": self.CELERY_MAX_TASKS_PER_CHILD,
            "worker_max_memory_per_child": self.CELERY_MAX_MEMORY_PER_CHILD,
            "result_expires": self.QUEUE_RESULT_TTL,
        }

    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration"""
        return {
            "per_hour": self.RATE_LIMIT_PER_HOUR,
            "burst": self.RATE_LIMIT_BURST,
            "global_per_minute": self.GLOBAL_RATE_LIMIT_PER_MINUTE,
        }

    # =====================================
    # Environment-specific Configurations
    # =====================================


# Create global settings instance
settings = Settings()



def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


# =====================================
# Environment-Specific Configurations
# =====================================


def get_development_settings() -> Settings:
    """Get settings optimized for development"""
    dev_settings = Settings(
        DEBUG=True,
        RELOAD=True,
        LOG_LEVEL="DEBUG",
        MOCK_GENERATION=True,
        MAX_CONCURRENT_TASKS=2,
        CELERY_WORKER_CONCURRENCY=1,
        REDIS_DB=1,  # Use different DB for development
    )
    return dev_settings


def get_production_settings() -> Settings:
    """Get settings optimized for production"""
    prod_settings = Settings(
        DEBUG=False,
        RELOAD=False,
        LOG_LEVEL="INFO",
        MOCK_GENERATION=False,
        MAX_CONCURRENT_TASKS=8,
        CELERY_WORKER_CONCURRENCY=4,
        ENABLE_METRICS=True,
        AUTO_CLEANUP_ENABLED=True,
    )
    return prod_settings


def get_testing_settings() -> Settings:
    """Get settings optimized for testing"""
    test_settings = Settings(
        TESTING=True,
        DEBUG=True,
        LOG_LEVEL="WARNING",
        MOCK_GENERATION=True,
        MAX_CONCURRENT_TASKS=1,
        REDIS_DB=2,  # Use different DB for testing
        QUEUE_RESULT_TTL=60,  # Short TTL for tests
        RATE_LIMIT_PER_HOUR=1000,  # Higher limit for tests
    )
    return test_settings


# =====================================
# Configuration Helpers
# =====================================


def get_settings_for_environment(env: str = None) -> Settings:  # type: ignore
    """Get settings based on environment"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")

    if env == "production":
        return get_production_settings()
    elif env == "testing":
        return get_testing_settings()
    else:
        return get_development_settings()


def validate_settings(
    settings_obj: Settings, *, create_dirs: bool = False, check_redis: bool = False
) -> bool:
    """Validate settings configuration (optionally creating directories/checking Redis)."""
    try:
        # Check required directories
        for path in [
            settings_obj.MODEL_BASE_PATH,
            settings_obj.OUTPUT_BASE_PATH,
            settings_obj.CACHE_BASE_PATH,
        ]:
            if create_dirs and not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            elif not path.exists():
                print(f"Warning: expected directory does not exist: {path}")

        # Check Redis connection (optional)
        if check_redis and settings_obj.get_redis_url():
            try:
                import redis  # type: ignore
            except Exception:
                print("Warning: redis client not installed; skipping Redis validation")
                return True

            client = redis.from_url(settings_obj.get_redis_url())
            client.ping()
            client.close()

        # Check GPU availability if specified
        if settings_obj.DEVICE == "cuda":
            import torch

            if not torch.cuda.is_available():
                print("Warning: CUDA specified but not available, falling back to CPU")
                settings_obj.DEVICE = "cpu"

        return True

    except Exception as e:
        print(f"Settings validation error: {e}")
        return False


# Avoid side effects on import; validation can be run explicitly.
if os.getenv("VALIDATE_SETTINGS_ON_IMPORT", "false").lower() == "true":
    if not validate_settings(settings, create_dirs=True, check_redis=False):
        print("Warning: Some settings validation checks failed")

# Export commonly used settings
__all__ = [
    "Settings",
    "settings",
    "get_development_settings",
    "get_production_settings",
    "get_testing_settings",
    "get_settings_for_environment",
    "validate_settings",
]

if __name__ == "__main__":
    """Main entry point for configuration validation"""
    print("🔧 SD Multi-Modal Platform Configuration")
    print(f"Device: {settings.DEVICE}")
    print(f"Primary Model: {settings.PRIMARY_MODEL}")
    print(f"Use SDPA: {settings.USE_SDPA}")
    print(f"Enable xformers: {settings.ENABLE_XFORMERS}")
    print(f"Model Path: {settings.get_model_path()}")
    print(f"Output Path: {settings.OUTPUT_PATH}")
