# app/services/generation/img2img_service.py
import logging
import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import torch
from PIL import Image
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
)

from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_error()

from app.config import settings
from utils.logging_utils import get_logger
from utils.image_utils import ImageProcessor
from utils.file_utils import FileManager
from utils.metadata_utils import ImageMetadata, MetadataManager

logger = logging.getLogger(__name__)


class Img2ImgService:
    """
    Image-to-image generation service with multi-model support

    Supports:
    - Stable Diffusion 1.5 img2img
    - Stable Diffusion XL img2img
    - ControlNet integration
    - Batch processing
    - Memory optimization
    """

    def __init__(self):
        self.is_initialized = False
        self.current_model = None
        self.pipeline: Optional[
            Union[StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline]
        ] = None
        self.device = settings.DEVICE
        self.torch_dtype = getattr(torch, settings.TORCH_DTYPE)

        # Service utilities
        self.image_processor = ImageProcessor()
        self.file_manager = FileManager()
        self.metadata_manager = MetadataManager()

        # Performance tracking
        self.total_generations = 0
        self.total_time = 0.0
        self.startup_time = None

    async def initialize(self, model_id: Optional[str] = None):
        """Initialize the img2img service with specified model"""
        try:
            start_time = time.time()
            logger.info("🎨 Initializing Img2Img service...")

            if StableDiffusionImg2ImgPipeline is None:
                logger.error("Diffusers library not available")
                raise ImportError("Please install diffusers: pip install diffusers")

            # Determine model to load
            target_model = model_id or settings.PRIMARY_SD_MODEL
            await self._load_model(target_model)

            # Apply optimizations
            await self._apply_optimizations()

            # Warm up with a test generation
            if not settings.MOCK_GENERATION:
                await self._warmup_model()

            self.startup_time = time.time() - start_time
            self.is_initialized = True

            logger.info(f"✅ Img2Img service initialized in {self.startup_time:.2f}s")
            logger.info(f"📋 Model: {self.current_model}, Device: {self.device}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Img2Img service: {e}")
            raise

    async def _load_model(self, model_id: str):
        """Load specified img2img model"""
        try:
            logger.info(f"📦 Loading img2img model: {model_id}")

            # Model configuration
            model_configs = {
                "sd-1.5": {
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "pipeline_class": StableDiffusionImg2ImgPipeline,
                    "requires_safety_checker": True,
                },
                "sdxl-base": {
                    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "pipeline_class": StableDiffusionXLImg2ImgPipeline,
                    "requires_safety_checker": False,
                },
                "sdxl-refiner": {
                    "model_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
                    "pipeline_class": StableDiffusionXLImg2ImgPipeline,
                    "requires_safety_checker": False,
                },
            }

            if model_id not in model_configs:
                logger.warning(f"Unknown model {model_id}, falling back to sd-1.5")
                model_id = "sd-1.5"

            config = model_configs[model_id]

            # Load pipeline
            pipeline_kwargs = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": True,
                "variant": "fp16" if self.torch_dtype == torch.float16 else None,
            }

            # Handle safety checker
            if not config["requires_safety_checker"]:
                pipeline_kwargs.update(
                    {"safety_checker": None, "requires_safety_checker": False}
                )

            self.pipeline = config["pipeline_class"].from_pretrained(
                config["model_id"], **pipeline_kwargs
            )

            # Move to device
            self.pipeline = self.pipeline.to(self.device)  # type: ignore
            self.current_model = model_id

            logger.info(f"✅ Model {model_id} loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {e}")
            raise

    async def _apply_optimizations(self):
        """Apply memory and performance optimizations"""
        try:
            logger.info("⚡ Applying optimizations...")

            if self.pipeline is None:
                return

            # Memory optimizations
            if settings.ENABLE_ATTENTION_SLICING:
                self.pipeline.enable_attention_slicing()
                logger.debug("✅ Attention slicing enabled")

            if settings.ENABLE_CPU_OFFLOAD and self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                logger.debug("✅ CPU offload enabled")

            # xFormers optimization
            if settings.ENABLE_XFORMERS:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.debug("✅ xFormers optimization enabled")
                except Exception as e:
                    logger.warning(f"xFormers not available: {e}")

            # Torch compile (experimental)
            if settings.ENABLE_TORCH_COMPILE:
                try:
                    self.pipeline.unet = torch.compile(
                        self.pipeline.unet, mode="reduce-overhead"
                    )
                    logger.debug("✅ Torch compile enabled")
                except Exception as e:
                    logger.warning(f"Torch compile failed: {e}")

            logger.info("✅ Optimizations applied")

        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")

    async def _warmup_model(self):
        """Warm up the model with a test generation"""
        try:
            logger.info("🔥 Warming up model...")

            # Create a small test image
            test_image = Image.new("RGB", (512, 512), color="white")

            # Run a quick generation
            with torch.no_grad():
                _ = self.pipeline(  # type: ignore
                    prompt="test",
                    image=test_image,
                    strength=0.5,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    output_type="pil",
                ).images[  # type: ignore
                    0
                ]

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ Model warmed up")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    async def generate_image(
        self,
        prompt: str,
        image: Union[Image.Image, str, Path],
        strength: float = 0.8,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate images using img2img pipeline

        Args:
            prompt: Text prompt for generation
            image: Input image (PIL Image, file path, or Path object)
            strength: How much to transform the input image (0.0-1.0)
            negative_prompt: Negative text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            width: Output width (optional, will resize input image)
            height: Output height (optional, will resize input image)
            user_id: User identifier for organization
            **kwargs: Additional pipeline parameters

        Returns:
            Dictionary containing generation results and metadata
        """
        try:
            generation_start = time.time()

            if not self.is_initialized:
                await self.initialize()

            # Validate and process input image
            input_image = await self._process_input_image(image, width, height)

            # Validate parameters
            strength = max(0.0, min(1.0, strength))
            num_inference_steps = max(1, min(100, num_inference_steps))
            guidance_scale = max(1.0, min(20.0, guidance_scale))
            num_images = max(1, min(4, num_images))

            # Set up random seed
            if seed is None or seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()  # type: ignore

            generator = torch.Generator(device=self.device).manual_seed(seed)  # type: ignore

            # Prepare generation parameters
            generation_params = {
                "prompt": prompt,
                "image": input_image,
                "strength": strength,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
                "generator": generator,
                "output_type": "pil",
                **kwargs,
            }

            # Remove None values
            generation_params = {
                k: v for k, v in generation_params.items() if v is not None
            }

            logger.info(
                f"🎨 Starting img2img generation: '{prompt[:50]}...' (strength: {strength})"
            )

            # Generate images
            with torch.no_grad():
                if settings.MOCK_GENERATION:
                    # Mock generation for testing
                    generated_images = [input_image.copy() for _ in range(num_images)]
                    await asyncio.sleep(0.1)  # Simulate processing time
                else:
                    result = self.pipeline(**generation_params)  # type: ignore
                    generated_images = self._extract_images_from_result(result)

            generation_time = time.time() - generation_start

            # Process and save results
            results = await self._process_generation_results(
                images=generated_images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=input_image,
                strength=strength,
                seed=seed,  # type: ignore
                generation_time=generation_time,
                user_id=user_id,
                generation_params=generation_params,
            )

            # Update statistics
            self.total_generations += len(generated_images)
            self.total_time += generation_time

            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"✅ Img2img generation completed in {generation_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"❌ Img2img generation failed: {e}")

            # GPU cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise

    async def _process_input_image(
        self,
        image: Union[Image.Image, str, Path],
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Image.Image:
        """Process and validate input image"""
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path)

            # Ensure RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if target dimensions provided
            if target_width and target_height:
                LANCZOS_FILTER = Image.Resampling.LANCZOS
                image = image.resize((target_width, target_height), LANCZOS_FILTER)
            else:
                # Ensure dimensions are multiple of 8 for stable diffusion
                width, height = image.size
                width = (width // 8) * 8
                height = (height // 8) * 8
                if (width, height) != image.size:
                    LANCZOS_FILTER = Image.Resampling.LANCZOS
                    image = image.resize((width, height), LANCZOS_FILTER)

            # Validate image size
            max_size = 2048
            if image.width > max_size or image.height > max_size:
                ratio = min(max_size / image.width, max_size / image.height)
                new_width = int(image.width * ratio // 8) * 8
                new_height = int(image.height * ratio // 8) * 8
                LANCZOS_FILTER = Image.Resampling.LANCZOS
                image = image.resize((new_width, new_height), LANCZOS_FILTER)
                logger.warning(
                    f"Image resized to {new_width}x{new_height} (max size exceeded)"
                )

            return image

        except Exception as e:
            logger.error(f"Failed to process input image: {e}")
            raise

    def _extract_images_from_result(self, result: Any) -> List[Image.Image]:
        """Extract PIL images from pipeline result"""
        try:
            if hasattr(result, "images") and result.images:
                return [img for img in result.images if isinstance(img, Image.Image)]
            elif isinstance(result, list):
                return [img for img in result if isinstance(img, Image.Image)]
            elif isinstance(result, Image.Image):
                return [result]
            else:
                logger.warning(f"Unexpected result type: {type(result)}")
                return []

        except Exception as e:
            logger.error(f"Failed to extract images from result: {e}")
            return []

    async def _process_generation_results(
        self,
        images: List[Image.Image],
        prompt: str,
        negative_prompt: str,
        input_image: Image.Image,
        strength: float,
        seed: int,
        generation_time: float,
        user_id: Optional[str],
        generation_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process and save generation results"""
        try:
            task_id = f"img2img_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

            # Save images and create metadata
            saved_images = []

            for i, image in enumerate(images):
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"img2img_{timestamp}_{seed}_{i:02d}.png"

                # Save image
                image_path = await self.file_manager.save_image(
                    image=image, filename=filename, subfolder="img2img", user_id=user_id
                )

                # Create metadata
                metadata = ImageMetadata(
                    filename=filename,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model=self.current_model,
                    width=image.width,
                    height=image.height,
                    seed=seed,
                    steps=generation_params.get("num_inference_steps", 25),
                    cfg_scale=generation_params.get("guidance_scale", 7.5),
                    strength=strength,
                    generation_time=generation_time,
                    task_id=task_id,
                    task_type="img2img",
                    user_id=user_id,
                    input_image_size=f"{input_image.width}x{input_image.height}",
                    additional_params={
                        k: v
                        for k, v in generation_params.items()
                        if k not in ["prompt", "image", "generator", "negative_prompt"]
                    },
                )

                # Save metadata
                metadata_path = await self.metadata_manager.save_metadata(
                    metadata=metadata, task_id=task_id
                )

                saved_images.append(
                    {
                        "image_path": str(image_path),
                        "image_url": f"/outputs/img2img/{filename}",
                        "metadata_path": str(metadata_path),
                        "width": image.width,
                        "height": image.height,
                    }
                )

            # Calculate VRAM usage
            vram_used = None
            if torch.cuda.is_available():
                vram_used = f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"

            return {
                "success": True,
                "task_id": task_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "input_image_size": f"{input_image.width}x{input_image.height}",
                "strength": strength,
                "parameters": {
                    "model": self.current_model,
                    "steps": generation_params.get("num_inference_steps", 25),
                    "cfg_scale": generation_params.get("guidance_scale", 7.5),
                    "seed": seed,
                    "num_images": len(images),
                },
                "result": {
                    "images": saved_images,
                    "image_count": len(saved_images),
                    "generation_time": round(generation_time, 2),
                    "model_used": self.current_model,
                    "vram_used": vram_used,
                    "device": self.device,
                },
                "metadata": {
                    "task_type": "img2img",
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                },
            }

        except Exception as e:
            logger.error(f"Failed to process generation results: {e}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        avg_time = self.total_time / max(1, self.total_generations)

        return {
            "service": "img2img",
            "initialized": self.is_initialized,
            "current_model": self.current_model,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "statistics": {
                "total_generations": self.total_generations,
                "total_time": round(self.total_time, 2),
                "average_time": round(avg_time, 2),
                "startup_time": round(self.startup_time or 0, 2),
            },
            "memory": {
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_allocated": (
                    f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                    if torch.cuda.is_available()
                    else None
                ),
                "gpu_memory_reserved": (
                    f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }

    async def cleanup(self):
        """Clean up resources and memory"""
        try:
            logger.info("🧹 Cleaning up Img2Img service...")

            if self.pipeline is not None:
                # Move pipeline to CPU to free GPU memory
                if hasattr(self.pipeline, "to"):
                    self.pipeline.to("cpu")

                del self.pipeline
                self.pipeline = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self.is_initialized = False
            self.current_model = None

            logger.info("✅ Img2Img service cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def switch_model(self, model_id: str):
        """Switch to a different model"""
        try:
            if self.current_model == model_id:
                logger.info(f"Model {model_id} already loaded")
                return

            logger.info(f"🔄 Switching from {self.current_model} to {model_id}")

            # Cleanup current model
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load new model
            await self._load_model(model_id)
            await self._apply_optimizations()

            logger.info(f"✅ Successfully switched to {model_id}")

        except Exception as e:
            logger.error(f"Failed to switch to model {model_id}: {e}")
            raise


# Global service instance
_img2img_service: Optional[Img2ImgService] = None


async def get_img2img_service() -> Img2ImgService:
    """Get global img2img service instance (singleton pattern)"""
    global _img2img_service
    if _img2img_service is None:
        _img2img_service = Img2ImgService()
        await _img2img_service.initialize()
    return _img2img_service


async def cleanup_img2img_service():
    """Cleanup global img2img service"""
    global _img2img_service
    if _img2img_service is not None:
        await _img2img_service.cleanup()
        _img2img_service = None
