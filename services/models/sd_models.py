# services/models/sd_models.py
"""
SD Multi-Modal Platform - Model Manager
This module manages the AI model lifecycle, including loading, unloading, and preloading models.
"""

import logging
import time
from typing import Optional, Dict, Any

import torch

from app.config import settings


logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the lifecycle of AI models for the SD Multi-Modal Platform."""

    def __init__(self):
        self.is_initialized = False
        self.startup_time = None

    async def warm_up(self):
        """Preload models and perform initial checks."""
        start_time = time.time()

        try:
            logger.info("🔥 Starting model warm-up...")
            # Basic device validation
            if settings.DEVICE == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA not available but DEVICE=cuda")
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                logger.info(f"📊 GPU Memory: {gpu_memory:.1f}GB")

                if gpu_memory < 6:
                    logger.warning("⚠️  Low GPU memory, consider using CPU offload")

            # Check model path
            model_path = settings.get_model_path()
            logger.info(f"📂 Model path: {model_path}")

            # Initialize model loading logic here
            self.is_initialized = True
            self.startup_time = time.time() - start_time

            logger.info(
                f"✅ Model manager warm-up completed ({self.startup_time:.2f}s)"
            )

        except Exception as exc:
            logger.error(f"❌ Model warm-up failed: {exc}")
            raise

    async def cleanup(self):
        """Clean up resources and unload models."""
        logger.info("🧹 Model manager cleanup...")

        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✅ CUDA cache cleared")

        self.is_initialized = False
        logger.info("✅ Model manager cleanup completed")

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the model manager."""
        return {
            "initialized": self.is_initialized,
            "startup_time": self.startup_time,
            "device": settings.DEVICE,
            "primary_model": settings.PRIMARY_MODEL,
        }


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """取得模型管理器實例"""
    global _model_manager

    if _model_manager is None:
        _model_manager = ModelManager()

    return _model_manager
