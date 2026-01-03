"""
Unified model cache management with device and precision settings.

This module is used by the service-layer model managers under `services/`.
It must respect the storage rules in `~/Desktop/data_model_structure.md`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from app.config import settings

logger = logging.getLogger(__name__)


class ModelCache:
    """Centralized in-process cache with device + dtype configuration."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self.device = self._setup_device()
        self.dtype = self._setup_dtype()

    def _setup_device(self) -> torch.device:
        device_setting = getattr(settings, "DEVICE", "auto")
        if device_setting == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_setting

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            device = "cpu"

        logger.info("Using device: %s", device)
        return torch.device(device)

    def _setup_dtype(self) -> torch.dtype:
        precision = getattr(settings, "PRECISION", "float32")
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = precision_map.get(precision, torch.float32)

        if dtype == torch.float16 and self.device.type == "cpu":
            logger.warning("float16 on CPU is limited; using float32 instead")
            dtype = torch.float32

        logger.info("Using precision: %s", dtype)
        return dtype

    def setup_memory_optimizations(self) -> None:
        # Placeholder hook for model-specific optimizations.
        # (e.g. attention slicing, CPU offload) are applied inside each pipeline.
        return

    def get_cache_key(self, model_type: str, model_name: str, **kwargs) -> str:
        parts = [model_type, model_name]
        for key, value in sorted(kwargs.items()):
            parts.append(f"{key}={value}")
        return ":".join(parts)

    def get(self, cache_key: str) -> Optional[Any]:
        return self._cache.get(cache_key)

    def set(self, cache_key: str, model: Any) -> None:
        self._cache[cache_key] = model

    def clear(self, cache_key: Optional[str] = None) -> None:
        if cache_key:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info("Cleared cache entry: %s", cache_key)
            return

        self._cache.clear()
        logger.info("Cleared all model cache entries")


model_cache = ModelCache()

