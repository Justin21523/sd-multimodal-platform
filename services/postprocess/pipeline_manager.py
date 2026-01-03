# services/postprocess/pipeline_manager.py
"""
Post-processing Pipeline Manager
Manages chained AI model processing: upscale + face restoration + custom filters
"""

import asyncio
import gc
import torch
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import logging

# Optional postprocess dependencies (keep this module importable without them).
try:
    from realesrgan import RealESRGANer  # type: ignore
    from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore

    REALESRGAN_AVAILABLE = True
except Exception:  # pragma: no cover
    RealESRGANer = None  # type: ignore
    RRDBNet = None  # type: ignore
    REALESRGAN_AVAILABLE = False

try:
    from gfpgan import GFPGANer  # type: ignore

    GFPGAN_AVAILABLE = True
except Exception:  # pragma: no cover
    GFPGANer = None  # type: ignore
    GFPGAN_AVAILABLE = False

try:
    from codeformer import CodeFormer  # type: ignore

    CODEFORMER_AVAILABLE = True
except Exception:  # pragma: no cover
    CodeFormer = None  # type: ignore
    CODEFORMER_AVAILABLE = False

from app.config import settings
from utils.logging_utils import get_generation_logger
from utils.image_utils import optimize_image, resize_image
from utils.file_utils import save_generation_output
from utils.metadata_utils import save_generation_metadata

logger = logging.getLogger(__name__)


class PostprocessStep:
    """Individual post-processing step definition"""

    def __init__(self, step_type: str, model_name: str, params: Dict[str, Any] = None):  # type: ignore[no-untyped-def]
        self.step_type = step_type  # 'upscale', 'face_restore', 'denoise'
        self.model_name = model_name  # 'real-esrgan-x4', 'gfpgan', etc.
        self.params = params or {}
        self.processing_time = 0.0

    def __repr__(self):
        return f"PostprocessStep({self.step_type}:{self.model_name})"


class PostprocessPipeline:
    """Chained post-processing pipeline with resource management"""

    def __init__(self):
        self.steps: List[PostprocessStep] = []
        self.loaded_models: Dict[str, Any] = {}
        self.total_vram_used = 0
        self.pipeline_id = None

    def add_step(
        self, step_type: str, model_name: str, **params
    ) -> "PostprocessPipeline":
        """Add processing step to pipeline (fluent interface)"""
        step = PostprocessStep(step_type, model_name, params)
        self.steps.append(step)
        return self

    async def execute(self, input_image: Image.Image, task_id: str) -> Dict[str, Any]:
        """Execute complete pipeline on input image"""

        gen_logger = get_generation_logger("postprocess", "pipeline")
        start_time = time.time()

        current_image = input_image.copy()
        step_results = []

        try:
            for i, step in enumerate(self.steps):
                step_start = time.time()

                gen_logger.info(f"Executing step {i+1}/{len(self.steps)}: {step}")

                # Load model if not cached
                if step.model_name not in self.loaded_models:
                    await self._load_model(step.step_type, step.model_name)

                # Execute processing step
                processed_image = await self._execute_step(step, current_image, task_id)

                step.processing_time = time.time() - step_start
                current_image = processed_image

                step_results.append(
                    {
                        "step": step.step_type,
                        "model": step.model_name,
                        "processing_time": step.processing_time,
                        "image_size": f"{current_image.width}x{current_image.height}",
                    }
                )

                gen_logger.info(
                    f"Step completed in {step.processing_time:.2f}s",
                    extra={"processing_time": step.processing_time},
                )

        except Exception as e:
            gen_logger.error(f"Pipeline execution failed: {str(e)}")
            raise

        finally:
            # Cleanup loaded models if VRAM pressure
            await self._cleanup_if_needed()

        total_time = time.time() - start_time

        result = {
            "processed_image": current_image,
            "steps_executed": len(self.steps),
            "step_details": step_results,
            "total_processing_time": total_time,
            "vram_used": self._get_vram_usage(),
        }

        gen_logger.info(
            f"Pipeline completed in {total_time:.2f}s",
            extra={"generation_time": total_time, "vram_used": result["vram_used"]},
        )

        return result

    async def _load_model(self, step_type: str, model_name: str):
        """Load post-processing model with memory management"""

        if step_type == "upscale":
            if "real-esrgan" in model_name:
                model = await self._load_realesrgan(model_name)
            else:
                raise ValueError(f"Unsupported upscale model: {model_name}")

        elif step_type == "face_restore":
            if "gfpgan" in model_name:
                model = await self._load_gfpgan(model_name)
            elif "codeformer" in model_name:
                model = await self._load_codeformer(model_name)
            else:
                raise ValueError(f"Unsupported face restoration model: {model_name}")

        else:
            raise ValueError(f"Unknown step type: {step_type}")

        self.loaded_models[model_name] = model
        logger.info(f"Loaded model: {model_name}")

    async def _load_realesrgan(self, model_name: str):
        """Load Real-ESRGAN upscaling model"""
        try:
            if not REALESRGAN_AVAILABLE or RealESRGANer is None or RRDBNet is None:
                raise ImportError(
                    "Real-ESRGAN not installed (missing `realesrgan`/`basicsr` and dependencies)."
                )
            # Model configuration mapping
            model_configs = {
                "real-esrgan-x4": {
                    "model_path": Path(settings.MODELS_PATH)
                    / "upscale"
                    / "RealESRGAN_x4plus.pth",
                    "scale": 4,
                    "arch": RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=4,
                    ),
                }
            }

            config = model_configs.get(model_name)
            if not config:
                raise ValueError(f"Unknown Real-ESRGAN model: {model_name}")

            model_path = config["model_path"]
            if not model_path.exists():
                legacy_path = (
                    Path(settings.MODELS_PATH)
                    / "upscale"
                    / "real-esrgan"
                    / "experiments"
                    / "pretrained_models"
                    / model_path.name
                )
                if legacy_path.exists():
                    model_path = legacy_path

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            upsampler = RealESRGANer(
                scale=config["scale"],
                model_path=str(model_path),
                model=config["arch"],
                device=settings.DEVICE,
                half=settings.TORCH_DTYPE == "float16",
            )

            return upsampler

        except ImportError:
            raise ImportError("Real-ESRGAN not installed. Run: pip install realesrgan basicsr")

    async def _load_gfpgan(self, model_name: str):
        """Load GFPGAN face restoration model"""
        try:
            if not GFPGAN_AVAILABLE or GFPGANer is None:
                raise ImportError("GFPGAN not installed (missing `gfpgan` and dependencies).")
            model_path = Path(settings.MODELS_PATH) / "face-restore" / "GFPGANv1.4.pth"
            if not model_path.exists():
                legacy_path = (
                    Path(settings.MODELS_PATH)
                    / "face-restore"
                    / "gfpgan"
                    / "experiments"
                    / "pretrained_models"
                    / "GFPGANv1.4.pth"
                )
                if legacy_path.exists():
                    model_path = legacy_path

            if not model_path.exists():
                raise FileNotFoundError(f"GFPGAN model not found: {model_path}")

            face_enhancer = GFPGANer(
                model_path=str(model_path),
                upscale=1,  # Don't upscale, just restore
                arch="clean",
                channel_multiplier=2,
                device=settings.DEVICE,
            )

            return face_enhancer

        except ImportError:
            raise ImportError("GFPGAN not installed. Run: pip install gfpgan")

    async def _load_codeformer(self, model_name: str):
        """Load CodeFormer face restoration model"""
        try:
            if not CODEFORMER_AVAILABLE or CodeFormer is None:
                raise ImportError("CodeFormer not installed (missing `codeformer` and dependencies).")
            model_path = Path(settings.MODELS_PATH) / "face-restore" / "codeformer.pth"
            if not model_path.exists():
                legacy_path = (
                    Path(settings.MODELS_PATH)
                    / "postprocess"
                    / "codeformer"
                    / "weights"
                    / "CodeFormer"
                    / "codeformer.pth"
                )
                if legacy_path.exists():
                    model_path = legacy_path

            if not model_path.exists():
                raise FileNotFoundError(f"CodeFormer model not found: {model_path}")

            # CodeFormer implementation would go here
            # This is a placeholder for the actual integration
            return None

        except ImportError:
            raise ImportError("CodeFormer not installed")

    async def _execute_step(
        self, step: PostprocessStep, image: Image.Image, task_id: str
    ) -> Image.Image:
        """Execute individual processing step"""

        model = self.loaded_models[step.model_name]

        if step.step_type == "upscale":
            return await self._upscale_image(model, image, step.params)
        elif step.step_type == "face_restore":
            return await self._restore_faces(model, image, step.params)
        else:
            raise ValueError(f"Unknown step type: {step.step_type}")

    async def _upscale_image(
        self, upsampler, image: Image.Image, params: Dict[str, Any]
    ) -> Image.Image:
        """Execute Real-ESRGAN upscaling"""

        # Convert PIL to OpenCV format
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Run upscaling
        with torch.inference_mode():
            upscaled_bgr, _ = upsampler.enhance(image_bgr)

        # Convert back to PIL
        upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        upscaled_image = Image.fromarray(upscaled_rgb)

        return upscaled_image

    async def _restore_faces(
        self, face_enhancer, image: Image.Image, params: Dict[str, Any]
    ) -> Image.Image:
        """Execute GFPGAN face restoration"""

        # Convert PIL to OpenCV format
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Extract parameters
        strength = params.get("strength", 1.0)  # 0.0-1.0

        # Run face restoration
        with torch.inference_mode():
            _, _, restored_bgr = face_enhancer.enhance(
                image_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=strength,
            )

        # Convert back to PIL
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        restored_image = Image.fromarray(restored_rgb)

        return restored_image

    async def _cleanup_if_needed(self):
        """Clean up models if VRAM usage is high"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > 10.0:  # If using more than 10GB
                await self._unload_all_models()

    async def _unload_all_models(self):
        """Unload all cached models to free VRAM"""
        for model_name in list(self.loaded_models.keys()):
            del self.loaded_models[model_name]

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info("Unloaded all post-processing models")

    def _get_vram_usage(self) -> str:
        """Get current VRAM usage string"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"{allocated:.2f}GB / {reserved:.2f}GB"
        return "N/A (CPU mode)"


# Factory function for pipeline creation
def create_standard_pipeline() -> PostprocessPipeline:
    """Create standard upscale + face restoration pipeline"""
    return (
        PostprocessPipeline()
        .add_step("upscale", "real-esrgan-x4", scale=4)
        .add_step("face_restore", "gfpgan", strength=0.8)
    )


def create_fast_pipeline() -> PostprocessPipeline:
    """Create fast face restoration only pipeline"""
    return PostprocessPipeline().add_step("face_restore", "gfpgan", strength=0.6)


def create_quality_pipeline() -> PostprocessPipeline:
    """Create high-quality pipeline with multiple steps"""
    return (
        PostprocessPipeline()
        .add_step("upscale", "real-esrgan-x4", scale=4)
        .add_step("face_restore", "gfpgan", strength=1.0)
    )


# Global pipeline manager instance
_pipeline_manager = None


def get_pipeline_manager() -> PostprocessPipeline:
    """Get global pipeline manager instance (singleton)"""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PostprocessPipeline()
    return _pipeline_manager
