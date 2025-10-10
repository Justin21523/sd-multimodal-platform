# app/services/models/model_cache.py
"""
Unified model cache management with device and precision settings
"""

import torch
import logging
from typing import Dict, Any, Optional
from app.config import settings
from app.shared_cache import shared_cache

logger = logging.getLogger(__name__)


class ModelCache:
    """Centralized model cache with device and precision management"""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self.device = self._setup_device()
        self.dtype = self._setup_dtype()

    def _setup_device(self) -> torch.device:
        """Setup computation device based on config"""
        if settings.DEVICE == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = settings.DEVICE

        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

        logger.info(f"Using device: {device}")
        return torch.device(device)

    def _setup_dtype(self) -> torch.dtype:
        """Setup precision dtype based on config"""
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        dtype = precision_map.get(settings.PRECISION, torch.float32)

        # Validate dtype compatibility
        if dtype == torch.float16 and self.device.type == "cpu":
            logger.warning("float16 not well supported on CPU, using float32")
            dtype = torch.float32

        logger.info(f"Using precision: {dtype}")
        return dtype

    def setup_memory_optimizations(self):
        """Apply memory optimizations based on config"""
        if settings.ENABLE_ATTENTION_SLICING and self.device.type == "cuda":
            try:
                # This would be implemented in specific model classes
                logger.info("Attention slicing enabled")
            except Exception as e:
                logger.warning(f"Failed to enable attention slicing: {e}")

    def get_cache_key(self, model_type: str, model_name: str, **kwargs) -> str:
        """Generate cache key for model"""
        key_parts = [model_type, model_name]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return ":".join(key_parts)

    def get(self, cache_key: str) -> Optional[Any]:
        """Get model from cache"""
        return self._cache.get(cache_key)

    def set(self, cache_key: str, model: Any):
        """Store model in cache"""
        self._cache[cache_key] = model

    def clear(self, cache_key: str = None):
        """Clear cache or specific entry"""
        if cache_key:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info(f"Cleared cache entry: {cache_key}")
        else:
            self._cache.clear()
            logger.info("Cleared all model cache entries")


# Global model cache instance
model_cache = ModelCache()
