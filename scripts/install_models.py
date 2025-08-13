#!/usr/bin/env python3
# scripts/install_models.py
"""
Model Download and Installation Script for SD Multi-Modal Platform
Downloads and sets up required AI models for Phase 3 implementation.
"""

import os
import logging
from contextlib import asynccontextmanager
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from huggingface_hub import snapshot_download, login
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Model configurations for Phase 3
MODEL_CONFIGS = {
    "sdxl-base": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_class": StableDiffusionXLPipeline,
        "local_path": "sdxl/sdxl-base",
        "vram_requirement": "8GB",
        "recommended_resolution": "1024x1024",
        "description": "High-quality photorealistic generation, best for commercial/advertising use",
    },
    "sd-1.5": {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "pipeline_class": StableDiffusionPipeline,
        "local_path": "stable-diffusion/sd-1.5",
        "vram_requirement": "4GB",
        "recommended_resolution": "512x512",
        "description": "Classic SD model, excellent LoRA ecosystem, anime/character generation",
    },
    "sd-2.1": {
        "repo_id": "stabilityai/stable-diffusion-2-1",
        "pipeline_class": StableDiffusionPipeline,
        "local_path": "stable-diffusion/sd-2.1",
        "vram_requirement": "6GB",
        "recommended_resolution": "768x768",
        "description": "Improved version of SD with better quality and composition",
    },
}


class ModelInstaller:
    """Manages model download and installation process."""

    def __init__(self):
        self.base_path = Path(settings.OUTPUT_PATH).parent / "models"
        self.base_path.mkdir(parents=True, exist_ok=True)
        setup_logging()

    async def check_system_requirements(self) -> Dict[str, Any]:
        """Check system compatibility and requirements."""
        logger.info("Checking system requirements...")

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_name = "CPU"
            total_vram = 0

        system_info = {
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram, 1),
            "pytorch_version": torch.__version__,
            "device": settings.DEVICE,
        }

        logger.info(f"System info: {system_info}")
        return system_info

    async def download_model(
        self, model_name: str, force_redownload: bool = False
    ) -> bool:
        """Download a specific model with progress tracking."""
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            return False

        config = MODEL_CONFIGS[model_name]
        local_path = self.base_path / config["local_path"]

        # Check if already exists
        if local_path.exists() and not force_redownload:
            logger.info(f"Model {model_name} already exists at {local_path}")
            return True

        logger.info(f"Downloading {model_name} from {config['repo_id']}...")
        logger.info(f"Target path: {local_path}")
        logger.info(f"VRAM requirement: {config['vram_requirement']}")

        try:
            # Create target directory
            local_path.mkdir(parents=True, exist_ok=True)

            # Download model
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            logger.info(f"✅ Successfully downloaded {model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {str(e)}")
            return False

    async def verify_model(self, model_name: str) -> bool:
        """Verify model integrity and loadability."""
        if model_name not in MODEL_CONFIGS:
            return False

        config = MODEL_CONFIGS[model_name]
        local_path = self.base_path / config["local_path"]

        if not local_path.exists():
            logger.error(f"Model path does not exist: {local_path}")
            return False

        logger.info(f"Verifying {model_name}...")

        try:
            # Try to load pipeline
            pipeline_class = config["pipeline_class"]
            pipeline = pipeline_class.from_pretrained(
                str(local_path),
                torch_dtype=(
                    torch.float16 if settings.DEVICE == "cuda" else torch.float32
                ),
                use_safetensors=True,
            )

            # Basic validation
            if hasattr(pipeline, "unet") and hasattr(pipeline, "vae"):
                logger.info(f"✅ Model {model_name} verification passed")
                del pipeline  # Free memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return True
            else:
                logger.error(f"❌ Model {model_name} missing required components")
                return False

        except Exception as e:
            logger.error(f"❌ Model {model_name} verification failed: {str(e)}")
            return False

    async def install_models(
        self, models: List[str], verify: bool = True
    ) -> Dict[str, bool]:
        """Install multiple models with verification."""
        results = {}

        logger.info(f"Installing models: {models}")

        for model_name in models:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing model: {model_name}")
            logger.info(f"{'='*50}")

            # Download
            download_success = await self.download_model(model_name)
            if not download_success:
                results[model_name] = False
                continue

            # Verify if requested
            if verify:
                verify_success = await self.verify_model(model_name)
                results[model_name] = verify_success
            else:
                results[model_name] = True

        return results

    def list_available_models(self) -> None:
        """Print available models and their info."""
        print("\n" + "=" * 80)
        print("AVAILABLE MODELS FOR SD MULTI-MODAL PLATFORM")
        print("=" * 80)

        for model_name, config in MODEL_CONFIGS.items():
            print(f"\n🤖 {model_name.upper()}")
            print(f"   Repository: {config['repo_id']}")
            print(f"   VRAM Requirement: {config['vram_requirement']}")
            print(f"   Recommended Resolution: {config['recommended_resolution']}")
            print(f"   Description: {config['description']}")

            # Check if installed
            local_path = self.base_path / config["local_path"]
            status = "✅ INSTALLED" if local_path.exists() else "❌ NOT INSTALLED"
            print(f"   Status: {status}")

        print("\n" + "=" * 80)


async def main():
    """Main installation script entry point."""
    # parser = argparse.ArgumentParser(
    #    description="Install models for SD Multi-Modal Platform"
    # )
    # parser.add_argument(
    #    "--models",
    #    nargs="+",
    #    choices=list(MODEL_CONFIGS.keys()) + ["all"],
    #    default=["sdxl-base"],
    #    help="Models to install (default: sdxl-base)",
    # )
    # parser.add_argument("--list", action="store_true", help="List available models")
    # parser.add_argument(
    #    "--verify", action="store_true", help="Verify models after download"
    # )
    # parser.add_argument(
    #    "--force", action="store_true", help="Force redownload even if model exists"
    # )
    # parser.add_argument(
    #    "--check-requirements",
    #    action="store_true",
    #    help="Check system requirements only",
    # )

    # args = parser.parse_args()

    installer = ModelInstaller()

    # List models and exit
    # if args.list:
    #    installer.list_available_models()
    #    return
    installer.list_available_models()

    # Check system first
    system_info = await installer.check_system_requirements()
    if not system_info["cuda_available"] and settings.DEVICE == "cuda":
        logger.warning(
            "⚠️  CUDA not available but device set to 'cuda'. Consider using CPU mode."
        )
    print(f"\nSystem Requirements Check:")
    print(f"CUDA Available: {system_info['cuda_available']}")
    print(f"GPU: {system_info['gpu_name']}")
    print(f"Total VRAM: {system_info['total_vram_gb']}GB")
    print(f"PyTorch Version: {system_info['pytorch_version']}")

    # Install models
    models_to_install = list(MODEL_CONFIGS.keys())

    # Install
    results = await installer.install_models(models_to_install, verify=True)

    # Print summary
    print(f"\n{'='*50}")
    print("INSTALLATION SUMMARY")
    print(f"{'='*50}")

    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name}: {status}")

    successful_installs = sum(results.values())
    total_installs = len(results)
    print(
        f"\nTotal: {successful_installs}/{total_installs} models installed successfully"
    )

    if successful_installs > 0:
        print(f"\n🎉 Ready for Phase 3! You can now start the application:")
        print(f"   python scripts/start_phase3.py")


if __name__ == "__main__":
    asyncio.run(main())
