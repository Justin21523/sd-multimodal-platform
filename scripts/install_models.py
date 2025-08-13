#!/usr/bin/env python3
# scripts/install_models.py
"""
Enhanced model installation script for SD Multi-Modal Platform Phase 4
Supports base models, ControlNet, and automatic dependency management.
"""

import os
import logging
from contextlib import asynccontextmanager
import sys
import time
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

from huggingface_hub import snapshot_download, hf_hub_download
from app.config import settings
from utils.file_utils import ensure_directory, get_file_size, get_directory_size
from utils.logging_utils import setup_logging

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

# Model configurations
BASE_MODELS = {
    "sdxl-base": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "description": "Stable Diffusion XL Base - High quality photoreal generation",
        "size_estimate": "12GB",
        "required_vram": "12GB+",
        "optimal_for": ["photography", "commercial", "high-quality"],
    },
    "sd-1.5": {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "description": "Stable Diffusion 1.5 - Anime/character generation, LoRA compatible",
        "size_estimate": "4GB",
        "required_vram": "6GB+",
        "optimal_for": ["anime", "characters", "LoRA", "fast generation"],
    },
    "sd-2.1": {
        "repo_id": "stabilityai/stable-diffusion-2-1",
        "description": "Stable Diffusion 2.1 - Balanced versatile model",
        "size_estimate": "5GB",
        "required_vram": "8GB+",
        "optimal_for": ["general purpose", "balanced quality"],
    },
}

CONTROLNET_MODELS = {
    "canny": {
        "repo_id": "lllyasviel/sd-controlnet-canny",
        "description": "Canny edge detection for precise line control",
        "size_estimate": "1.4GB",
        "use_cases": ["line art", "architectural drawings", "precise edges"],
    },
    "openpose": {
        "repo_id": "lllyasviel/sd-controlnet-openpose",
        "description": "Human pose control for character positioning",
        "size_estimate": "1.4GB",
        "use_cases": ["human poses", "character positioning", "dance", "sports"],
    },
    "depth": {
        "repo_id": "lllyasviel/sd-controlnet-depth",
        "description": "Depth map control for 3D scene structure",
        "size_estimate": "1.4GB",
        "use_cases": ["3D scenes", "depth perception", "layered composition"],
    },
    "scribble": {
        "repo_id": "lllyasviel/sd-controlnet-scribble",
        "description": "Scribble/sketch control for rough guidance",
        "size_estimate": "1.4GB",
        "use_cases": ["sketches", "rough layouts", "quick concepts"],
    },
    "mlsd": {
        "repo_id": "lllyasviel/sd-controlnet-mlsd",
        "description": "Straight line detection for architectural control",
        "size_estimate": "1.4GB",
        "use_cases": ["architecture", "interior design", "geometric shapes"],
    },
    "normal": {
        "repo_id": "lllyasviel/sd-controlnet-normal",
        "description": "Surface normal control for detailed textures",
        "size_estimate": "1.4GB",
        "use_cases": ["surface details", "texture control", "material rendering"],
    },
}

# SDXL ControlNet models (separate because they're different)
SDXL_CONTROLNET_MODELS = {
    "canny-sdxl": {
        "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
        "description": "SDXL Canny ControlNet for high-res edge control",
        "size_estimate": "2.5GB",
        "use_cases": ["high-res line art", "detailed architectural drawings"],
    },
    "openpose-sdxl": {
        "repo_id": "thibaud/controlnet-openpose-sdxl-1.0",
        "description": "SDXL OpenPose ControlNet for detailed human poses",
        "size_estimate": "2.5GB",
        "use_cases": ["high-res human poses", "detailed character work"],
    },
    "depth-sdxl": {
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
        "description": "SDXL Depth ControlNet for detailed 3D scenes",
        "size_estimate": "2.5GB",
        "use_cases": ["high-res 3D scenes", "detailed depth control"],
    },
}

# Post-processing models
POSTPROCESS_MODELS = {
    "real-esrgan-x4": {
        "repo_id": "ai-forever/Real-ESRGAN",
        "files": ["RealESRGAN_x4plus.pth"],
        "description": "4x upscaling for photos and art",
        "size_estimate": "64MB",
    },
    "real-esrgan-anime": {
        "repo_id": "ai-forever/Real-ESRGAN",
        "files": ["RealESRGAN_x4plus_anime_6B.pth"],
        "description": "4x upscaling optimized for anime",
        "size_estimate": "18MB",
    },
    "gfpgan": {
        "repo_id": "TencentARC/GFPGAN",
        "files": ["GFPGANv1.4.pth"],
        "description": "Face restoration and enhancement",
        "size_estimate": "348MB",
    },
}


class ModelInstaller:
    """Manages model download and installation process."""

    def __init__(self):
        self.base_path = Path(settings.OUTPUT_PATH).parent / "models"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.downloaded_models = []
        self.failed_downloads = []
        self.total_size_downloaded = 0
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

    async def install_base_model(self, model_id: str) -> bool:
        """Install a base Stable Diffusion model"""
        if model_id not in BASE_MODELS:
            logger.error(f"Unknown base model: {model_id}")
            return False

        model_config = BASE_MODELS[model_id]
        repo_id = model_config["repo_id"]

        # Determine target directory
        target_dir = self.base_path / "stable-diffusion" / model_id

        logger.info(f"📥 Installing base model: {model_id}")
        logger.info(f"   Repository: {repo_id}")
        logger.info(f"   Target: {target_dir}")
        logger.info(f"   Estimated size: {model_config['size_estimate']}")

        try:
            # Download model
            start_time = time.time()

            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            download_time = time.time() - start_time
            actual_size = get_directory_size(target_dir)

            logger.info(f"✅ Downloaded {model_id} in {download_time:.1f}s")
            logger.info(f"   Actual size: {actual_size / 1024**3:.2f}GB")

            self.downloaded_models.append(
                {
                    "model_id": model_id,
                    "type": "base",
                    "path": str(target_dir),
                    "size_bytes": actual_size,
                    "download_time": download_time,
                }
            )
            self.total_size_downloaded += actual_size

            return True

        except Exception as e:
            logger.error(f"❌ Failed to download {model_id}: {str(e)}")
            self.failed_downloads.append({"model_id": model_id, "error": str(e)})
            return False

    async def install_controlnet_model(
        self, controlnet_id: str, for_sdxl: bool = False
    ) -> bool:
        """Install a ControlNet model"""
        if for_sdxl:
            if controlnet_id not in SDXL_CONTROLNET_MODELS:
                logger.error(f"Unknown SDXL ControlNet: {controlnet_id}")
                return False
            model_config = SDXL_CONTROLNET_MODELS[controlnet_id]
            target_dir = self.base_path / "controlnet" / "sdxl" / controlnet_id
        else:
            if controlnet_id not in CONTROLNET_MODELS:
                logger.error(f"Unknown ControlNet: {controlnet_id}")
                return False
            model_config = CONTROLNET_MODELS[controlnet_id]
            target_dir = self.base_path / "controlnet" / "sd" / controlnet_id

        repo_id = model_config["repo_id"]

        logger.info(f"📥 Installing ControlNet: {controlnet_id}")
        logger.info(f"   Repository: {repo_id}")
        logger.info(f"   Target: {target_dir}")
        logger.info(f"   Estimated size: {model_config['size_estimate']}")

        try:
            start_time = time.time()

            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            download_time = time.time() - start_time
            actual_size = get_directory_size(target_dir)

            logger.info(
                f"✅ Downloaded ControlNet {controlnet_id} in {download_time:.1f}s"
            )

            self.downloaded_models.append(
                {
                    "model_id": controlnet_id,
                    "type": "controlnet_sdxl" if for_sdxl else "controlnet",
                    "path": str(target_dir),
                    "size_bytes": actual_size,
                    "download_time": download_time,
                }
            )
            self.total_size_downloaded += actual_size

            return True

        except Exception as e:
            logger.error(f"❌ Failed to download ControlNet {controlnet_id}: {str(e)}")
            self.failed_downloads.append({"model_id": controlnet_id, "error": str(e)})
            return False

    async def install_postprocess_model(self, model_id: str) -> bool:
        """Install post-processing models (upscalers, face restoration)"""
        if model_id not in POSTPROCESS_MODELS:
            logger.error(f"Unknown post-process model: {model_id}")
            return False

        model_config = POSTPROCESS_MODELS[model_id]
        repo_id = model_config["repo_id"]
        files = model_config.get("files", [])

        # Determine target directory
        if "esrgan" in model_id:
            target_dir = self.base_path / "upscale" / "real-esrgan"
        elif "gfpgan" in model_id:
            target_dir = self.base_path / "face-restore" / "gfpgan"
        else:
            target_dir = self.base_path / "postprocess" / model_id

        ensure_directory(target_dir)

        logger.info(f"📥 Installing post-process model: {model_id}")
        logger.info(f"   Repository: {repo_id}")
        logger.info(f"   Target: {target_dir}")

        try:
            start_time = time.time()
            total_size = 0

            if files:
                # Download specific files
                for filename in files:
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=target_dir,
                        local_dir_use_symlinks=False,
                    )
                    total_size += get_file_size(file_path)
            else:
                # Download entire repository
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                total_size = get_directory_size(target_dir)

            download_time = time.time() - start_time

            logger.info(f"✅ Downloaded {model_id} in {download_time:.1f}s")
            logger.info(f"   Size: {total_size / 1024**2:.1f}MB")

            self.downloaded_models.append(
                {
                    "model_id": model_id,
                    "type": "postprocess",
                    "path": str(target_dir),
                    "size_bytes": total_size,
                    "download_time": download_time,
                }
            )
            self.total_size_downloaded += total_size

            return True

        except Exception as e:
            logger.error(f"❌ Failed to download {model_id}: {str(e)}")
            self.failed_downloads.append({"model_id": model_id, "error": str(e)})
            return False

    def create_directory_structure(self):
        """Create necessary directory structure"""
        directories = [
            "stable-diffusion",
            "controlnet/sd",
            "controlnet/sdxl",
            "lora",
            "vae",
            "upscale/real-esrgan",
            "face-restore/gfpgan",
            "face-restore/codeformer",
            "postprocess",
        ]

        for directory in directories:
            dir_path = self.base_path / directory
            ensure_directory(dir_path)
            logger.info(f"📁 Created directory: {dir_path}")

    def print_summary(self):
        """Print installation summary"""
        print("\n" + "=" * 80)
        print("🎉 MODEL INSTALLATION SUMMARY")
        print("=" * 80)

        if self.downloaded_models:
            print(f"\n✅ Successfully downloaded {len(self.downloaded_models)} models:")
            for model in self.downloaded_models:
                size_mb = model["size_bytes"] / 1024**2
                print(f"   • {model['model_id']} ({model['type']}) - {size_mb:.1f}MB")

        if self.failed_downloads:
            print(f"\n❌ Failed downloads ({len(self.failed_downloads)}):")
            for failure in self.failed_downloads:
                print(f"   • {failure['model_id']}: {failure['error']}")

        total_gb = self.total_size_downloaded / 1024**3
        print(f"\n📊 Total downloaded: {total_gb:.2f}GB")
        print(f"📁 Models location: {self.base_path}")

    # old phase 3 download function
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
    """Main installation function"""
    parser = argparse.ArgumentParser(
        description="Install AI models for SD Multi-Modal Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/install_models.py --base sdxl-base sd-1.5
  python scripts/install_models.py --controlnet canny openpose depth
  python scripts/install_models.py --controlnet-sdxl canny-sdxl depth-sdxl
  python scripts/install_models.py --postprocess real-esrgan-x4 gfpgan
  python scripts/install_models.py --all
  python scripts/install_models.py --minimal
        """,
    )

    # Model selection arguments
    parser.add_argument(
        "--base",
        nargs="*",
        choices=list(BASE_MODELS.keys()),
        help="Install base models",
    )
    parser.add_argument(
        "--controlnet",
        nargs="*",
        choices=list(CONTROLNET_MODELS.keys()),
        help="Install ControlNet models for SD 1.5/2.1",
    )
    parser.add_argument(
        "--controlnet-sdxl",
        nargs="*",
        choices=list(SDXL_CONTROLNET_MODELS.keys()),
        help="Install ControlNet models for SDXL",
    )
    parser.add_argument(
        "--postprocess",
        nargs="*",
        choices=list(POSTPROCESS_MODELS.keys()),
        help="Install post-processing models",
    )

    # Convenience options
    parser.add_argument(
        "--all",
        action="store_true",
        help="Install all available models (requires significant disk space)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Install minimal set: SDXL base + basic ControlNet",
    )
    parser.add_argument(
        "--recommended",
        action="store_true",
        help="Install recommended set for most use cases",
    )

    # Options
    parser.add_argument(
        "--list", action="store_true", help="List all available models and exit"
    )
    parser.add_argument(
        "--check-space", action="store_true", help="Check available disk space"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO")

    if args.list:
        print_available_models()
        return

    if args.check_space:
        check_disk_space()
        return

    # Initialize installer
    installer = ModelInstaller()
    installer.create_directory_structure()

    # Determine what to install
    base_models = []
    controlnet_models = []
    controlnet_sdxl_models = []
    postprocess_models = []

    if args.all:
        base_models = list(BASE_MODELS.keys())
        controlnet_models = list(CONTROLNET_MODELS.keys())
        controlnet_sdxl_models = list(SDXL_CONTROLNET_MODELS.keys())
        postprocess_models = list(POSTPROCESS_MODELS.keys())
    elif args.minimal:
        base_models = ["sdxl-base"]
        controlnet_models = ["canny", "openpose"]
        postprocess_models = ["real-esrgan-x4"]
    elif args.recommended:
        base_models = ["sdxl-base", "sd-1.5"]
        controlnet_models = ["canny", "openpose", "depth"]
        controlnet_sdxl_models = ["canny-sdxl", "openpose-sdxl"]
        postprocess_models = ["real-esrgan-x4", "real-esrgan-anime", "gfpgan"]
    else:
        # Use individual arguments
        base_models = args.base or []
        controlnet_models = args.controlnet or []
        controlnet_sdxl_models = args.controlnet_sdxl or []
        postprocess_models = args.postprocess or []

    # Validate that something was requested
    total_models = (
        len(base_models)
        + len(controlnet_models)
        + len(controlnet_sdxl_models)
        + len(postprocess_models)
    )
    if total_models == 0:
        print("❌ No models specified for installation.")
        print("Use --help to see available options or --list to see all models.")
        return

    # Show installation plan
    print("📋 INSTALLATION PLAN")
    print("=" * 50)
    if base_models:
        print(f"Base Models ({len(base_models)}): {', '.join(base_models)}")
    if controlnet_models:
        print(
            f"ControlNet SD ({len(controlnet_models)}): {', '.join(controlnet_models)}"
        )
    if controlnet_sdxl_models:
        print(
            f"ControlNet SDXL ({len(controlnet_sdxl_models)}): {', '.join(controlnet_sdxl_models)}"
        )
    if postprocess_models:
        print(
            f"Post-process ({len(postprocess_models)}): {', '.join(postprocess_models)}"
        )

    # Calculate estimated space
    estimated_space = calculate_estimated_space(
        base_models, controlnet_models, controlnet_sdxl_models, postprocess_models
    )
    print(f"\n💾 Estimated download size: {estimated_space:.1f}GB")

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be downloaded")
        return

    # Confirm installation
    try:
        confirm = input("\n❓ Proceed with installation? [Y/n]: ").strip().lower()
        if confirm and confirm not in ["y", "yes"]:
            print("Installation cancelled.")
            return
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        return

    print("\n🚀 Starting model installation...")
    start_time = time.time()

    # Install base models
    for model_id in base_models:
        print(f"\n📦 Installing base model: {model_id}")
        success = await installer.install_base_model(model_id)
        if not success:
            print(f"⚠️  Continuing despite failure...")

    # Install ControlNet models
    for model_id in controlnet_models:
        print(f"\n📦 Installing ControlNet: {model_id}")
        success = await installer.install_controlnet_model(model_id, for_sdxl=False)
        if not success:
            print(f"⚠️  Continuing despite failure...")

    # Install SDXL ControlNet models
    for model_id in controlnet_sdxl_models:
        print(f"\n📦 Installing SDXL ControlNet: {model_id}")
        success = await installer.install_controlnet_model(model_id, for_sdxl=True)
        if not success:
            print(f"⚠️  Continuing despite failure...")

    # Install post-processing models
    for model_id in postprocess_models:
        print(f"\n📦 Installing post-process model: {model_id}")
        success = await installer.install_postprocess_model(model_id)
        if not success:
            print(f"⚠️  Continuing despite failure...")

    total_time = time.time() - start_time

    # Show final summary
    installer.print_summary()
    print(f"\n⏱️  Total installation time: {total_time/60:.1f} minutes")

    # Post-installation setup
    print("\n🔧 POST-INSTALLATION SETUP")
    print("=" * 50)
    print("1. Update your .env file with model paths:")
    print(f"   SD_MODEL_PATH={installer.base_path}/stable-diffusion")
    print(f"   CONTROLNET_PATH={installer.base_path}/controlnet")
    print(f"   UPSCALE_MODEL_PATH={installer.base_path}/upscale")
    print("")
    print("2. Restart the application to load new models")
    print("")
    print("3. Test installation with:")
    print("   python scripts/test_phase4.py")


def print_available_models():
    """Print all available models with details"""
    print("📚 AVAILABLE MODELS")
    print("=" * 80)

    print("\n🏗️  BASE MODELS:")
    for model_id, config in BASE_MODELS.items():
        print(f"  {model_id:15} - {config['description']}")
        print(
            f"                    Size: {config['size_estimate']}, VRAM: {config['required_vram']}"
        )
        print(f"                    Best for: {', '.join(config['optimal_for'])}")
        print()

    print("🎮 CONTROLNET MODELS (SD 1.5/2.1):")
    for model_id, config in CONTROLNET_MODELS.items():
        print(f"  {model_id:15} - {config['description']}")
        print(f"                    Size: {config['size_estimate']}")
        print(f"                    Use cases: {', '.join(config['use_cases'])}")
        print()

    print("🎮 CONTROLNET MODELS (SDXL):")
    for model_id, config in SDXL_CONTROLNET_MODELS.items():
        print(f"  {model_id:15} - {config['description']}")
        print(f"                    Size: {config['size_estimate']}")
        print(f"                    Use cases: {', '.join(config['use_cases'])}")
        print()

    print("🛠️  POST-PROCESSING MODELS:")
    for model_id, config in POSTPROCESS_MODELS.items():
        print(f"  {model_id:15} - {config['description']}")
        print(f"                    Size: {config['size_estimate']}")
        print()


def calculate_estimated_space(
    base_models, controlnet_models, controlnet_sdxl_models, postprocess_models
):
    """Calculate estimated download space in GB"""
    total_gb = 0.0

    # Size estimates in GB
    size_map = {
        # Base models
        "sdxl-base": 12.0,
        "sd-1.5": 4.0,
        "sd-2.1": 5.0,
        # ControlNet SD
        "canny": 1.4,
        "openpose": 1.4,
        "depth": 1.4,
        "scribble": 1.4,
        "mlsd": 1.4,
        "normal": 1.4,
        # ControlNet SDXL
        "canny-sdxl": 2.5,
        "openpose-sdxl": 2.5,
        "depth-sdxl": 2.5,
        # Post-processing
        "real-esrgan-x4": 0.064,
        "real-esrgan-anime": 0.018,
        "gfpgan": 0.348,
    }

    all_models = (
        base_models + controlnet_models + controlnet_sdxl_models + postprocess_models
    )

    for model in all_models:
        total_gb += size_map.get(model, 1.0)  # Default 1GB if unknown

    return total_gb


def check_disk_space():
    """Check available disk space"""
    import shutil

    try:
        model_path = Path(settings.SD_MODEL_PATH).parent
        total, used, free = shutil.disk_usage(model_path)

        print("💾 DISK SPACE ANALYSIS")
        print("=" * 50)
        print(f"Target location: {model_path}")
        print(f"Total space:     {total / 1024**3:.1f}GB")
        print(f"Used space:      {used / 1024**3:.1f}GB")
        print(f"Free space:      {free / 1024**3:.1f}GB")
        print()

        # Recommendations
        if free / 1024**3 > 50:
            print("✅ Sufficient space for --all option (50GB+ available)")
        elif free / 1024**3 > 20:
            print("✅ Sufficient space for --recommended option (20GB+ available)")
        elif free / 1024**3 > 15:
            print("⚠️  Limited space - consider --minimal option")
        else:
            print("❌ Insufficient space - free up disk space before installing")

    except Exception as e:
        print(f"❌ Could not check disk space: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⛔ Installation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Installation failed: {str(e)}")
        print(f"\n❌ Installation failed: {str(e)}")
        sys.exit(1)
