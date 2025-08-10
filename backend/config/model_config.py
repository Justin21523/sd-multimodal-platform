# backend/config/model_config.py
"""
Model Configuration and Registry

Centralizes model metadata and loading parameters.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from .settings import Settings


@dataclass
class ModelConfig:
    """Configuration for a specific model"""

    name: str
    path: str
    torch_dtype: str = "float16"
    revision: Optional[str] = None
    safety_checker: bool = False
    requires_safety_checker: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for diffusers loading"""
        config = {
            "torch_dtype": getattr(__import__("torch"), self.torch_dtype),
            "safety_checker": None if not self.safety_checker else "default",
            "requires_safety_checker": self.requires_safety_checker,
        }

        if self.revision:
            config["revision"] = self.revision

        return config


class ModelRegistry:
    """Registry of available models with their configurations"""

    MODELS = {
        "sd-1.5": ModelConfig(
            name="Stable Diffusion 1.5",
            path="runwayml/stable-diffusion-v1-5",
            torch_dtype="float16",
            safety_checker=False,
            requires_safety_checker=False,
        ),
        "sd-2.1": ModelConfig(
            name="Stable Diffusion 2.1",
            path="stabilityai/stable-diffusion-2-1",
            torch_dtype="float16",
            safety_checker=False,
            requires_safety_checker=False,
        ),
        "sdxl": ModelConfig(
            name="Stable Diffusion XL",
            path="stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype="float16",
            safety_checker=False,
            requires_safety_checker=False,
        ),
    }

    @classmethod
    def get_model_config(cls, model_id: str) -> ModelConfig:
        """Get model configuration by ID"""
        if model_id not in cls.MODELS:
            raise ValueError(
                f"Unknown model: {model_id}. Available: {list(cls.MODELS.keys())}"
            )
        return cls.MODELS[model_id]

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List all available models"""
        return {model_id: config.name for model_id, config in cls.MODELS.items()}


# Global settings instance
settings = Settings()
settings.create_directories()
