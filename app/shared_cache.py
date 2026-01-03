"""
Shared cache and storage path management.

This module MUST follow the workstation storage spec in:
  ~/Desktop/data_model_structure.md

- Caches live under /mnt/c/ai_cache
- Model weights live under /mnt/c/ai_models
- Large outputs/datasets live under /mnt/data
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Dict

logger = logging.getLogger(__name__)


class SharedCache:
    """
    Configure cache-related environment variables so large downloads do NOT end up
    under ~/.cache or the system disk by default.
    """

    def __init__(self):
        self.cache_root = os.getenv("AI_CACHE_ROOT", "/mnt/c/ai_cache")
        self.models_root = os.getenv("AI_MODELS_ROOT", "/mnt/c/ai_models")
        self.paths = self._build_paths()
        self._apply_and_create()

    def _build_paths(self) -> Dict[str, str]:
        hf_root = f"{self.cache_root}/huggingface"
        return {
            # Hugging Face caches
            "HF_HOME": hf_root,
            "TRANSFORMERS_CACHE": hf_root,
            "HF_DATASETS_CACHE": f"{hf_root}/datasets",
            "HUGGINGFACE_HUB_CACHE": f"{hf_root}/hub",
            # Torch cache
            "TORCH_HOME": f"{self.cache_root}/torch",
            # XDG cache base
            "XDG_CACHE_HOME": self.cache_root,
        }

    def _apply_and_create(self) -> None:
        for key, path in self.paths.items():
            os.environ.setdefault(key, path)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        pathlib.Path(self.models_root).mkdir(parents=True, exist_ok=True)

        logger.info(
            "✅ SharedCache ready",
            extra={"cache_root": self.cache_root, "models_root": self.models_root},
        )

    def get_model_path(self, relative_path: str) -> str:
        """
        Resolve a model path under /mnt/c/ai_models.

        Example:
          shared_cache.get_model_path(\"stable-diffusion/sd-1.5\")
        """
        return str(pathlib.Path(self.models_root) / relative_path)

    def get_cache_info(self) -> Dict[str, str]:
        return dict(self.paths)


shared_cache = SharedCache()
