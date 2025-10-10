"""
Shared cache management for AI models and datasets
"""

import os
import pathlib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SharedCache:
    """Manages shared cache directories for AI models"""

    def __init__(self):
        self.cache_root = os.getenv(
            "AI_CACHE_ROOT", "/mnt/c/AI_LLM_projects/ai_warehouse/cache"
        )
        self.paths = self._setup_cache_paths()
        self._create_directories()

    def _setup_cache_paths(self) -> Dict[str, str]:
        """Define all cache paths"""
        return {
            "HF_HOME": f"{self.cache_root}/hf",
            "TRANSFORMERS_CACHE": f"{self.cache_root}/hf/transformers",
            "HF_DATASETS_CACHE": f"{self.cache_root}/hf/datasets",
            "HUGGINGFACE_HUB_CACHE": f"{self.cache_root}/hf/hub",
            "TORCH_HOME": f"{self.cache_root}/torch",
            "MODELS_DIR": f"{self.cache_root}/models",
            "DATA_DIR": f"{self.cache_root}/data",
        }

    def _create_directories(self):
        """Create cache directories if they don't exist"""
        for key, path in self.paths.items():
            os.environ[key] = path
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Cache directory ready: {key}={path}")

        logger.info(f"✅ SharedCache initialized: {self.cache_root}")

    def get_model_path(self, model_name: str) -> str:
        """Get path for specific model"""
        return f"{self.paths['MODELS_DIR']}/{model_name}"

    def get_cache_info(self) -> Dict[str, str]:
        """Get cache information"""
        return self.paths


# Global cache instance
shared_cache = SharedCache()
