# backend/core/model_loader.py
"""
Model Loading and Management Utilities

Provides centralized model loading with caching and error recovery.
"""

import torch
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from huggingface_hub import snapshot_download
from backend.config.settings import Settings
from backend.config.model_config import ModelRegistry

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model downloading and loading operations"""

    def __init__(self):
        self.cache_dir = Path(Settings.sd_model_path).parent
        self.download_progress = {}

    def check_model_availability(self, model_id: str) -> Dict[str, Any]:
        """
        Check if model is available locally or needs download

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with availability status and details
        """
        try:
            model_config = ModelRegistry.get_model_config(model_id)
            local_path = self.cache_dir / model_id

            status = {
                "model_id": model_id,
                "name": model_config.name,
                "local_path": str(local_path),
                "available_locally": local_path.exists(),
                "hub_path": model_config.path,
            }

            if local_path.exists():
                # Calculate local model size
                size_bytes = sum(
                    f.stat().st_size for f in local_path.rglob("*") if f.is_file()
                )
                status["size_gb"] = size_bytes / (1024**3)
                status["files_count"] = len(list(local_path.rglob("*")))

            return status

        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return {"error": str(e)}

    def download_model_if_needed(self, model_id: str, force: bool = False) -> bool:
        """
        Download model if not available locally

        Args:
            model_id: Model identifier
            force: Force download even if model exists

        Returns:
            Success status
        """
        try:
            model_config = ModelRegistry.get_model_config(model_id)
            local_path = self.cache_dir / model_id

            # Check if download is needed
            if local_path.exists() and not force:
                logger.info(f"Model {model_id} already available locally")
                return True

            logger.info(f"Downloading model: {model_config.name}")

            # Create download directory
            local_path.mkdir(parents=True, exist_ok=True)

            # Download using HuggingFace Hub
            snapshot_download(
                repo_id=model_config.path,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            logger.info(f"✅ Model {model_id} downloaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to download model {model_id}: {e}")
            return False

    def get_memory_requirements(self, model_id: str) -> Dict[str, Any]:
        """
        Estimate memory requirements for model

        Args:
            model_id: Model identifier

        Returns:
            Memory requirement estimates
        """
        # Rough estimates based on model types
        memory_estimates = {
            "sd-1.5": {"vram_gb": 4.0, "ram_gb": 8.0},
            "sd-2.1": {"vram_gb": 5.0, "ram_gb": 10.0},
            "sdxl": {"vram_gb": 8.0, "ram_gb": 16.0},
        }

        return memory_estimates.get(model_id, {"vram_gb": 6.0, "ram_gb": 12.0})

    def verify_model_integrity(self, model_id: str) -> bool:
        """
        Verify downloaded model integrity

        Args:
            model_id: Model identifier

        Returns:
            Verification success status
        """
        try:
            local_path = self.cache_dir / model_id

            if not local_path.exists():
                return False

            # Check for essential files
            essential_files = [
                "model_index.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
                "text_encoder/pytorch_model.bin",
            ]

            for file_name in essential_files:
                file_path = local_path / file_name
                if not file_path.exists():
                    logger.warning(f"Missing essential file: {file_name}")
                    return False

            logger.info(f"Model {model_id} integrity verified")
            return True

        except Exception as e:
            logger.error(f"Model integrity check failed: {e}")
            return False


# Global instances
image_processor = ImageProcessor()
model_loader = ModelLoader()
