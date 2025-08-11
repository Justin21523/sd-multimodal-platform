# services/models/sd_models.py
"""
Complete Model Manager for SD Multi-Modal Platform Phase 3
Handles model loading, switching, memory management, and pipeline abstraction.
"""

import logging
import gc
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from PIL import Image

from app.config import settings
from services.generation.txt2img_service import Txt2ImgService
from utils.logging_utils import setup_logging, get_request_logger
from utils.attention_utils import setup_attention_processor, setup_memory_optimizations

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for available models and their configurations."""

    MODELS = {
        "sdxl-base": {
            "name": "Stable Diffusion XL Base",
            "pipeline_class": StableDiffusionXLPipeline,
            "local_path": "models/sdxl/sdxl-base",
            "default_resolution": (1024, 1024),
            "vram_requirement": 8,
            "strengths": ["photorealistic", "commercial", "advertising", "high-detail"],
            "use_cases": [
                "product photos",
                "portraits",
                "landscapes",
                "commercial art",
            ],
        },
        "sd-1.5": {
            "name": "Stable Diffusion v1.5",
            "pipeline_class": StableDiffusionPipeline,
            "local_path": "models/stable-diffusion/sd-1.5",
            "default_resolution": (512, 512),
            "vram_requirement": 4,
            "strengths": ["anime", "character", "lora-ecosystem", "fast"],
            "use_cases": [
                "anime art",
                "character design",
                "stylized art",
                "concept art",
            ],
        },
        "sd-2.1": {
            "name": "Stable Diffusion v2.1",
            "pipeline_class": StableDiffusionPipeline,
            "local_path": "models/stable-diffusion/sd-2.1",
            "default_resolution": (768, 768),
            "vram_requirement": 6,
            "strengths": ["improved-quality", "composition", "versatile"],
            "use_cases": ["general purpose", "improved quality", "better composition"],
        },
    }

    @classmethod
    def get_model_info(cls, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model configuration by ID."""
        return cls.MODELS.get(model_id)

    @classmethod
    def list_models(cls) -> List[str]:
        """Get list of available model IDs."""
        return list(cls.MODELS.keys())

    @classmethod
    def find_models_by_strength(cls, strength: str) -> List[str]:
        """Find models that excel in specific area."""
        matching_models = []
        for model_id, config in cls.MODELS.items():
            if strength.lower() in config.get("strengths", []):
                matching_models.append(model_id)
        return matching_models


class ModelManager:
    """
    Complete model management system for SD Multi-Modal Platform.
    Handles loading, switching, memory management, and pipeline abstraction.
    """

    def __init__(self):
        self.is_initialized: bool = False
        self.current_pipeline: Optional[
            Union[StableDiffusionXLPipeline, StableDiffusionPipeline]
        ] = None
        self.current_model_id: Optional[str] = None
        self.startup_time: float = 0.0
        self.last_optimization_info: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}  # For future multi-model caching
        self.base_models_path = Path(settings.OUTPUT_PATH).parent / "models"

    async def initialize(self, model_id: Optional[str] = None) -> bool:
        """
        Initialize model manager and load primary model.

        Args:
            model_id: Specific model to load, or None for default (based on PRIMARY_MODEL setting)

        Returns:
            bool: True if initialization successful
        """
        start_time = time.time()
        logger.info("Initializing ModelManager...")

        try:
            # Determine which model to load
            target_model = model_id or settings.PRIMARY_MODEL
            logger.info(f"Target model: {target_model}")

            # Validate model exists
            model_info = ModelRegistry.get_model_info(target_model)
            if not model_info:
                logger.error(f"Unknown model: {target_model}")
                return False

            # Check if model files exist
            model_path = self.base_models_path / model_info["local_path"]
            if not model_path.exists():
                logger.error(f"Model files not found at: {model_path}")
                logger.info("Run 'python scripts/install_models.py' to download models")
                return False

            # Load the model
            success = await self._load_model(target_model)
            if not success:
                return False

            self.startup_time = time.time() - start_time
            self.is_initialized = True

            logger.info(
                f"✅ ModelManager initialized successfully in {self.startup_time:.2f}s"
            )
            logger.info(f"Current model: {self.current_model_id}")
            logger.info(f"Optimization info: {self.last_optimization_info}")

            return True

        except Exception as e:
            logger.error(f"❌ ModelManager initialization failed: {str(e)}")
            return False

    async def _load_model(self, model_id: str) -> bool:
        """Load a specific model with optimizations."""
        logger.info(f"Loading model: {model_id}")

        try:
            model_info = ModelRegistry.get_model_info(model_id)
            model_path = self.base_models_path / model_info["local_path"]  # type: ignore
            pipeline_class = model_info["pipeline_class"]  # type: ignore

            # Unload current model if exists
            if self.current_pipeline is not None:
                await self._unload_current_model()

            # Load pipeline
            logger.info(f"Loading pipeline from: {model_path}")
            pipeline = pipeline_class.from_pretrained(
                str(model_path),
                torch_dtype=settings.get_torch_dtype(),
                use_safetensors=True,
                variant="fp16" if settings.TORCH_DTYPE == "float16" else None,
            )

            # Move to device
            pipeline = pipeline.to(settings.DEVICE)

            # Apply optimizations
            optimization_info = self._apply_optimizations(pipeline)

            # Set optimized scheduler
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )

            # Store loaded pipeline
            self.current_pipeline = pipeline
            self.current_model_id = model_id
            self.last_optimization_info = optimization_info

            logger.info(f"✅ Model {model_id} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {str(e)}")
            return False

    def _apply_optimizations(self, pipeline) -> Dict[str, Any]:
        """Apply memory and performance optimizations to pipeline."""
        optimization_info = {
            "attention_type": "unknown",
            "memory_optimizations": [],
            "compilation": False,
        }

        try:
            # Setup attention processor (RTX 5080 optimized)
            attention_type = setup_attention_processor(
                pipeline,
                prefer_xformers=settings.ENABLE_XFORMERS,
                force_sdpa=settings.USE_SDPA,
            )
            optimization_info["attention_type"] = attention_type

            # Memory optimizations
            memory_opts = setup_memory_optimizations(
                pipeline,
                attention_type=attention_type,
                enable_cpu_offload=settings.ENABLE_CPU_OFFLOAD,
                enable_attention_slicing=settings.USE_ATTENTION_SLICING,
            )
            optimization_info["memory_optimizations"] = memory_opts

            # Model compilation (PyTorch 2.0+)
            if hasattr(torch, "compile") and settings.ENABLE_MODEL_COMPILATION:
                try:
                    pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
                    optimization_info["compilation"] = True
                    logger.info("✅ Model compilation enabled")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")

        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")

        return optimization_info

    async def _unload_current_model(self) -> None:
        """Safely unload current model and free memory."""
        if self.current_pipeline is None:
            return

        logger.info(f"Unloading model: {self.current_model_id}")

        try:
            # Move to CPU to free GPU memory
            self.current_pipeline = self.current_pipeline.to("cpu")

            # Clear references
            del self.current_pipeline
            self.current_pipeline = None
            self.current_model_id = None

            # Force garbage collection
            gc.collect()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("✅ Model unloaded and memory cleared")

        except Exception as e:
            logger.error(f"Error during model unloading: {e}")

    async def switch_model(self, model_id: str) -> bool:
        """
        Switch to a different model.

        Args:
            model_id: Target model ID

        Returns:
            bool: True if switch successful
        """
        if model_id == self.current_model_id:
            logger.info(f"Model {model_id} is already loaded")
            return True

        logger.info(f"Switching from {self.current_model_id} to {model_id}")

        # Load new model (this automatically unloads current)
        success = await self._load_model(model_id)
        if success:
            logger.info(f"✅ Successfully switched to model: {model_id}")
        else:
            logger.error(f"❌ Failed to switch to model: {model_id}")

        return success

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 0,
        height: int = 0,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
        num_images: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate images using current loaded model.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            width: Image width (None for model default)
            height: Image height (None for model default)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed (None for random)
            num_images: Number of images to generate

        Returns:
            Dict containing generation results and metadata
        """
        if not self.is_initialized or self.current_pipeline is None:
            raise RuntimeError("ModelManager not initialized or no model loaded")

        # Get model defaults if dimensions not specified
        model_info = ModelRegistry.get_model_info(self.current_model_id)  # type: ignore
        default_width, default_height = model_info["default_resolution"]  # type: ignore

        width = width or default_width
        height = height or default_height

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=settings.DEVICE).manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            generator = torch.Generator(device=settings.DEVICE).manual_seed(seed)

        generation_start = time.time()
        logger.info(f"Generating image with {self.current_model_id}: {prompt[:100]}...")

        try:
            # Record initial VRAM usage
            vram_before = 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_before = torch.cuda.memory_allocated() / (1024**3)

            # Generate images
            result = self.current_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_images,
                return_dict=True,
            )

            # Extract images safely using existing utility
            service = Txt2ImgService()
            images = service._extract_images_from_result(result)

            # Record final VRAM usage
            vram_after = 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_after = torch.cuda.memory_allocated() / (1024**3)

            generation_time = time.time() - generation_start

            generation_result = {
                "images": images,
                "metadata": {
                    "model_id": self.current_model_id,
                    "model_name": model_info["name"],  # type: ignore
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "num_images": len(images),
                    "generation_time": round(generation_time, 2),
                    "vram_used_gb": round(vram_after, 2),
                    "vram_delta_gb": round(vram_after - vram_before, 2),
                    "optimization_info": self.last_optimization_info,
                },
            }

            logger.info(f"✅ Generated {len(images)} images in {generation_time:.2f}s")
            logger.info(
                f"VRAM usage: {vram_after:.2f}GB (Δ{vram_after - vram_before:+.2f}GB)"
            )

            return generation_result

        except Exception as e:
            logger.error(f"❌ Image generation failed: {str(e)}")

            # Clear GPU cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise RuntimeError(f"Image generation failed: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """Get current model manager status."""
        status = {
            "is_initialized": self.is_initialized,
            "current_model_id": self.current_model_id,
            "startup_time": self.startup_time,
            "optimization_info": self.last_optimization_info,
            "available_models": ModelRegistry.list_models(),
        }

        # Add current model details
        if self.current_model_id:
            model_info = ModelRegistry.get_model_info(self.current_model_id)
            status["current_model_info"] = model_info

        # Add memory info
        if torch.cuda.is_available():
            status["vram_info"] = {
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                "total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                ),
            }

        return status

    async def cleanup(self) -> None:
        """Clean up resources and unload models."""
        logger.info("Cleaning up ModelManager...")

        if self.current_pipeline is not None:
            await self._unload_current_model()

        self.is_initialized = False
        self.model_cache.clear()

        logger.info("✅ ModelManager cleanup completed")


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance (singleton pattern)."""
    global _model_manager

    if _model_manager is None:
        _model_manager = ModelManager()

    return _model_manager
