# services/generation/txt2img_service.py
"""
SD Multi-Modal Platform - txt2img Generation Service
Phase 1: Text-to-Image Generation Service
"""

import logging
import time
import base64
import hashlib
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from io import BytesIO
import numpy as np

import torch
from PIL import Image
import diffusers
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler


from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)

from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)


from app.config import settings
from app.schemas.responses import ImageMetadata
from utils.image_utils import optimize_image, get_image_info
from utils.file_utils import ensure_directory, cleanup_old_files
from utils.metadata_utils import save_metadata_json


logger = logging.getLogger(__name__)


class Txt2ImgService:
    """Text-to-Image Generation Service"""

    def __init__(self):
        self.pipeline: Optional[
            Union[StableDiffusionXLPipeline, StableDiffusionPipeline]
        ] = None  # Type of the diffusion pipeline
        self.current_model = None
        self.device = settings.DEVICE
        self.torch_dtype = settings.get_torch_dtype()

        # Statistics tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.service_start_time = time.time()

        # Configuration flags
        self.enable_attention_slicing = settings.USE_ATTENTION_SLICING
        self.enable_cpu_offload = settings.ENABLE_CPU_OFFLOAD
        self.enable_xformers = settings.ENABLE_XFORMERS

        # Output directories
        self.output_dir = Path(settings.OUTPUT_PATH) / "txt2img"
        self.metadata_dir = Path(settings.OUTPUT_PATH) / "metadata"

        logger.info(f"🎨 Txt2ImgService initialized - Device: {self.device}")

    async def initialize(self):
        """Initialize the Txt2ImgService"""
        logger.info("🔧 Initializing txt2img service...")

        # Ensure output directories exist
        ensure_directory(self.output_dir)
        ensure_directory(self.metadata_dir)

        # Load the primary model
        try:
            await self._load_primary_model()
            logger.info("✅ Txt2ImgService initialization completed")
        except Exception as exc:
            logger.error(f"❌ Service initialization failed: {exc}")
            raise

    async def _load_primary_model(self):
        """Load the primary model based on configuration settings"""
        model_id = settings.PRIMARY_MODEL
        model_path = settings.get_model_path()

        logger.info(f"📦 Loading primary model: {model_id} from {model_path}")

        start_time = time.time()

        try:
            if model_id == "sdxl-base":
                # Load SDXL model
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True,
                    variant="fp16" if self.torch_dtype == torch.float16 else None,
                )

            elif model_id == "sd-1.5":
                # Load SD 1.5 model
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_path, torch_dtype=self.torch_dtype, use_safetensors=True
                )

            else:
                raise ValueError(f"Unsupported model: {model_id}")

            # Set the scheduler
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )

            # Move pipeline to the specified device
            self.pipeline = self.pipeline.to(self.device)

            # Enable optimizations
            if self.enable_attention_slicing:
                self.pipeline.enable_attention_slicing()
                logger.info("✅ Attention slicing enabled")

            if self.enable_cpu_offload and self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                logger.info("✅ CPU offload enabled")

            if self.enable_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ xFormers optimization enabled")
                except Exception as e:
                    logger.warning(f"⚠️  xFormers not available: {e}")

            # Compile the model if available
            try:
                if hasattr(torch, "compile"):
                    self.pipeline.unet = torch.compile(self.pipeline.unet)
                    logger.info("✅ Model compilation enabled")
            except Exception as e:
                logger.warning(f"⚠️  Model compilation failed: {e}")

            self.current_model = model_id
            load_time = time.time() - start_time

            logger.info(
                f"✅ Model loaded successfully: {model_id}",
                extra={
                    "model": model_id,
                    "load_time": load_time,
                    "device": self.device,
                    "dtype": str(self.torch_dtype),
                },
            )

        except Exception as exc:
            logger.error(f"❌ Failed to load model {model_id}: {exc}")
            raise

    async def ensure_model_loaded(self, model_id: Optional[str] = None) -> str:
        """Ensure the specified model is loaded, or load the primary model if not specified"""
        target_model = model_id or settings.PRIMARY_MODEL

        if self.pipeline is None or self.current_model != target_model:
            await self._load_primary_model()

        return self.current_model or ""

    def _extract_images_from_result(self, result: Any) -> List[Image.Image]:
        """
        Extract images from the pipeline result, handling various types safely.

        Args:
            result: The output from the diffusion pipeline, which can be of various types.

        Returns:
            List[PIL.Image]: A list of PIL Image objects extracted from the result.
        """
        images = []

        try:
            # Case 1: StableDiffusionPipelineOutput or StableDiffusionXLPipelineOutput
            if hasattr(result, "images") and result.images is not None:
                images = result.images
                logger.debug(
                    f"📋 Extracted {len(images)} images from .images attribute"
                )

            # Case 2: Direct list or tuple of images
            elif isinstance(result, (list, tuple)):
                images = list(result)
                logger.debug(f"📋 Got {len(images)} images from direct list")

            # Case 3: Single PIL Image
            elif isinstance(result, Image.Image):
                images = [result]
                logger.debug("📋 Got single PIL Image")

            # Case 4: numpy array
            elif isinstance(result, np.ndarray):
                if result.ndim == 4:  # Batch: (B, H, W, C)
                    images = [
                        Image.fromarray((img * 255).astype(np.uint8)) for img in result
                    ]
                elif result.ndim == 3:  # Single : (H, W, C)
                    images = [Image.fromarray((result * 255).astype(np.uint8))]
                logger.debug(f"📋 Converted {len(images)} images from numpy array")

            else:
                # Case 5: Unknown type, attempt to extract images
                logger.warning(f"⚠️  Unknown result type: {type(result)}")

                # Try to extract images from common attributes
                for attr_name in ["images", "image", "samples", "data"]:
                    if hasattr(result, attr_name):
                        attr_value = getattr(result, attr_name)
                        if attr_value is not None:
                            return self._extract_images_from_result(attr_value)

                # If we reach here, we have an unsupported type
                if hasattr(result, "__iter__"):
                    images = list(result)
                else:
                    raise ValueError(
                        f"Cannot extract images from result type: {type(result)}"
                    )

            # Validate and convert images to PIL format
            validated_images = []
            for i, img in enumerate(images):
                try:
                    if isinstance(img, Image.Image):
                        validated_images.append(img)
                    elif isinstance(img, np.ndarray):
                        # Make sure numpy array is in uint8 format
                        if img.dtype != np.uint8:
                            img = (
                                (img * 255).astype(np.uint8)
                                if img.max() <= 1.0
                                else img.astype(np.uint8)
                            )
                        validated_images.append(Image.fromarray(img))
                    elif torch.is_tensor(img):
                        # Convert PyTorch tensor to PIL Image
                        img_np = img.detach().cpu().numpy()
                        if img_np.dtype != np.uint8:
                            img_np = (
                                (img_np * 255).astype(np.uint8)
                                if img_np.max() <= 1.0
                                else img_np.astype(np.uint8)
                            )
                        validated_images.append(Image.fromarray(img_np))
                    else:
                        logger.warning(
                            f"⚠️  Skipping unsupported image type at index {i}: {type(img)}"
                        )

                except Exception as e:
                    logger.error(f"❌ Failed to process image at index {i}: {e}")
                    continue

            if not validated_images:
                raise ValueError("No valid images found in pipeline result")

            logger.info(
                f"✅ Successfully extracted {len(validated_images)} valid images"
            )
            return validated_images

        except Exception as exc:
            logger.error(f"❌ Failed to extract images from result: {exc}")
            logger.error(f"Result type: {type(result)}")
            if hasattr(result, "__dict__"):
                logger.error(f"Result attributes: {list(result.__dict__.keys())}")
            raise ValueError(f"Image extraction failed: {exc}")

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate an image based on the provided parameters.

        Returns:
            Dict containing:
            - image_path: Path to the generated image file
            - image: PIL Image object
            - generation_time: Time taken for generation (seconds)
            - vram_used: VRAM used during generation (if applicable)
            - model_hash: Hash of the model used for generation
        """

        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        logger.info(
            f"🎨 Starting image generation",
            extra={
                "prompt_length": len(prompt),
                "dimensions": f"{width}x{height}",
                "steps": num_inference_steps,
                "cfg": guidance_scale,
                "seed": seed,
            },
        )

        start_time = time.time()
        initial_memory = self._get_gpu_memory() if self.device == "cuda" else 0

        try:
            # Ensure model is loaded
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None

            # Prepare generation parameters
            generation_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }

            # Add any additional kwargs to the generation parameters
            logger.debug("🔄 Running diffusion pipeline...")

            with torch.inference_mode():
                raw_result = self.pipeline(**generation_params)

            # Extract images from the result
            images = self._extract_images_from_result(raw_result)

            if not images:
                raise ValueError("No images generated from pipeline")

            image = images[0]  # Use the first image if multiple are returned

            # Memory usage after generation
            peak_memory = self._get_gpu_memory() if self.device == "cuda" else 0
            vram_used = f"{peak_memory:.2f}GB" if peak_memory > 0 else "N/A"

            generation_time = time.time() - start_time

            # Save the generated image
            image_path = await self._save_image(image, prompt, seed)

            # Optimize the image
            self.generation_count += 1
            self.total_generation_time += generation_time

            # Save metadata
            file_info = get_image_info(image_path)  # type: ignore

            logger.info(
                f"✅ Image generation completed",
                extra={
                    "generation_time": generation_time,
                    "vram_used": vram_used,
                    "output_path": str(image_path),
                    "file_size": file_info.get("file_size"),
                },
            )

            return {
                "image_path": str(image_path),
                "image": image,
                "generation_time": generation_time,
                "vram_used": vram_used,
                "peak_memory": peak_memory,
                "model_hash": self._get_model_hash(),
                "file_size": file_info.get("file_size"),
                "success": True,
            }

        except Exception as exc:
            generation_time = time.time() - start_time

            logger.error(
                f"❌ Image generation failed: {exc}",
                extra={
                    "error_type": type(exc).__name__,
                    "generation_time": generation_time,
                    "prompt_length": len(prompt),
                },
                exc_info=True,
            )

            # Clean up resources if necessary
            if self.device == "cuda":
                torch.cuda.empty_cache()

            raise

    async def _save_image(self, image: Image.Image, prompt: str, seed: int) -> Path:
        """Save the generated image to the output directory with a unique filename."""

        # Generate a unique filename based on timestamp and prompt hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        filename = f"{timestamp}_{prompt_hash}_{seed}.png"

        output_path = self.output_dir / filename

        # Ensure the output directory exists
        optimized_image = optimize_image(image, quality=95, optimize=True)
        optimized_image.save(output_path, format="PNG", optimize=True)

        logger.debug(f"💾 Image saved: {output_path}")

        return output_path

    async def save_metadata(self, metadata: ImageMetadata, task_id: str) -> Path:
        """Save metadata to a JSON file in the metadata directory."""

        metadata_filename = f"{task_id}_metadata.json"
        metadata_path = self.metadata_dir / metadata_filename

        # Ensure the metadata directory exists
        save_metadata_json(metadata.to_dict(), metadata_path)

        logger.debug(f"📋 Metadata saved: {metadata_path}")

        return metadata_path

    async def get_image_base64(self, image_path: str) -> str:
        """Convert an image file to a base64-encoded string."""

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            base64_data = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"

        except Exception as exc:
            logger.error(f"Failed to encode image to base64: {exc}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Txt2ImgService."""

        uptime = time.time() - self.service_start_time
        avg_time = (
            self.total_generation_time / self.generation_count
            if self.generation_count > 0
            else 0
        )

        status = {
            "current_model": self.current_model,
            "model_loaded": self.pipeline is not None,
            "generation_count": self.generation_count,
            "avg_time": round(avg_time, 2),
            "uptime": round(uptime, 2),
            "recent_count": self.generation_count,  # Recent generations count
            "device": self.device,
        }

        # Add VRAM usage if applicable
        if self.device == "cuda" and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)

            status["memory_info"] = {
                "allocated_gb": round(memory_allocated, 2),
                "reserved_gb": round(memory_reserved, 2),
            }

        return status

    async def cleanup_old_files(self, days: int = 7):
        """Clean up old files in the output and metadata directories."""

        try:
            # Clean up output images
            cleanup_old_files(self.output_dir, days)

            # Clean up metadata files
            cleanup_old_files(self.metadata_dir, days)

            logger.info(f"🧹 Cleaned up files older than {days} days")

        except Exception as exc:
            logger.error(f"Cleanup failed: {exc}")

    def _get_gpu_memory(self) -> float:
        """Get the current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024**3)
        return 0.0

    def _get_model_hash(self) -> Optional[str]:
        """Get a hash of the current model for identification."""
        # If the pipeline is not loaded, return None
        return f"{self.current_model}_{settings.TORCH_DTYPE}"

    async def cleanup(self):
        """Clean up resources and unload the pipeline."""
        logger.info("🛑 Cleaning up txt2img service...")

        if self.pipeline is not None:
            # Clear the pipeline to free up memory
            del self.pipeline
            self.pipeline = None

            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("✅ Pipeline memory cleared")

        self.current_model = None

        logger.info("✅ Txt2ImgService cleanup completed")
