# backend/core/sd_pipeline.py
"""
Stable Diffusion Pipeline Core

Provides high-level interface for text-to-image generation with:
- Memory optimization for various hardware configurations
- Error handling and recovery mechanisms
- Support for multiple SD model variants
- Batch processing capabilities
"""

import torch
import gc
import logging
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
)
from diffusers.utils import logging as diffusers_logging

from backend.config.settings import Settings
from backend.config.model_config import ModelRegistry, ModelConfig

# Configure diffusers logging
diffusers_logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class StableDiffusionManager:
    """
    Central manager for Stable Diffusion operations

    Features:
    - Lazy model loading for memory efficiency
    - Multiple scheduler support for quality/speed trade-offs
    - Automatic memory management and cleanup
    - Comprehensive error handling
    """

    def __init__(self, model_id: str = "sd-1.5", device: str = None):
        self.model_id = model_id
        self.device = device or Settings.device
        self.pipeline = None
        self.model_config = ModelRegistry.get_model_config(model_id)

        # Performance optimization flags
        self.memory_optimized = False
        self.attention_slicing_enabled = False

        logger.info(
            f"Initializing SD Manager with model: {self.model_id} on device: {self.device}"
        )

    def _setup_memory_optimizations(self):
        """Configure memory optimizations based on available VRAM"""
        if self.pipeline is None:
            return

        try:
            # Enable CPU offload for large models
            if Settings.enable_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
                logger.info("✅ Enabled CPU offload")

            # Enable attention slicing for VRAM optimization
            if Settings.enable_attention_slicing:
                self.pipeline.enable_attention_slicing()
                self.attention_slicing_enabled = True
                logger.info("✅ Enabled attention slicing")

            # Enable memory efficient attention if xformers available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"xformers not available: {e}")

            self.memory_optimized = True

        except Exception as e:
            logger.error(f"Failed to setup memory optimizations: {e}")

    def load_pipeline(self, force_reload: bool = False) -> bool:
        """
        Load the Stable Diffusion pipeline with optimizations

        Args:
            force_reload: Force reload even if pipeline exists

        Returns:
            bool: Success status
        """
        if self.pipeline is not None and not force_reload:
            logger.info("Pipeline already loaded")
            return True

        try:
            logger.info(f"Loading pipeline: {self.model_config.name}")

            # Check if local model exists, otherwise use HuggingFace Hub
            local_path = Path(Settings.sd_model_path)
            model_path = (
                str(local_path) if local_path.exists() else self.model_config.path
            )

            logger.info(f"Model path: {model_path}")

            # Load pipeline with configuration
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_path, **self.model_config.to_dict()
            )

            # Setup optimized scheduler (DPM-Solver++ for speed)
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )

            # Move to target device
            self.pipeline = self.pipeline.to(self.device)

            # Apply memory optimizations
            self._setup_memory_optimizations()

            logger.info(f"✅ Pipeline loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load pipeline: {e}")
            self.pipeline = None
            return False

    def set_scheduler(self, scheduler_name: str = "dpm"):
        """
        Change the diffusion scheduler

        Args:
            scheduler_name: Scheduler type (dpm, euler_a, ddim)
        """
        if self.pipeline is None:
            logger.error("Pipeline not loaded")
            return False

        schedulers = {
            "dpm": DPMSolverMultistepScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "ddim": DDIMScheduler,
        }

        if scheduler_name not in schedulers:
            logger.error(f"Unknown scheduler: {scheduler_name}")
            return False

        try:
            SchedulerClass = schedulers[scheduler_name]
            self.pipeline.scheduler = SchedulerClass.from_config(
                self.pipeline.scheduler.config
            )
            logger.info(f"✅ Scheduler changed to: {scheduler_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to change scheduler: {e}")
            return False

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = None,
        height: int = None,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        seed: Optional[int] = None,
        batch_size: int = 1,
    ) -> List[Image.Image]:
        """
        Generate images from text prompts

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            width: Image width (default from settings)
            height: Image height (default from settings)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            batch_size: Number of images to generate

        Returns:
            List of PIL Images
        """
        # Ensure pipeline is loaded
        if self.pipeline is None:
            if not self.load_pipeline():
                raise RuntimeError("Failed to load pipeline")

        # Use defaults from settings
        width = width or Settings.default_width
        height = height or Settings.default_height
        num_inference_steps = num_inference_steps or Settings.default_steps
        guidance_scale = guidance_scale or Settings.default_guidance_scale

        # Validate batch size
        if batch_size > Settings.max_batch_size:
            logger.warning(
                f"Batch size {batch_size} exceeds max {Settings.max_batch_size}, clamping"
            )
            batch_size = Settings.max_batch_size

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        try:
            logger.info(f"Generating {batch_size} image(s): '{prompt[:50]}...'")

            with torch.inference_mode():
                # Clear CUDA cache before generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=batch_size,
                )

                images = result.images

                # Clear cache after generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                logger.info(f"✅ Generated {len(images)} image(s) successfully")
                return images

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    "CUDA out of memory. Try reducing batch_size or image dimensions"
                )
                # Attempt memory cleanup
                self._cleanup_memory()
            raise e
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise e

    def _cleanup_memory(self):
        """Aggressive memory cleanup for recovery"""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            logger.info("Memory cleanup completed")
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline"""
        if self.pipeline is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_id": self.model_id,
            "model_name": self.model_config.name,
            "device": self.device,
            "scheduler": self.pipeline.scheduler.__class__.__name__,
            "memory_optimized": self.memory_optimized,
            "attention_slicing": self.attention_slicing_enabled,
        }

    def unload_pipeline(self):
        """Unload pipeline to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._cleanup_memory()
            logger.info("Pipeline unloaded")


# Global SD manager instance
sd_manager = StableDiffusionManager()
