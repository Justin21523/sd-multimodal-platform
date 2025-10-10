# services/models/sd_models.py
"""
Complete Model Manager for SD Multi-Modal Platform Phase 3
Extended Stable Diffusion model management with img2img, inpaint, and auto-selection
"""

import logging
import gc
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
from PIL import Image

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
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
    StableDiffusionXLInpaintPipeline,
)
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
)

from app.config import settings
from app.shared_cache import shared_cache
from .model_cache import model_cache
from services.generation.txt2img_service import Txt2ImgService
from utils.attention_utils import setup_attention_processor
from utils.logging_utils import get_generation_logger
from utils.image_utils import pil_image_to_base64


logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for available models and their configurations."""

    MODELS = {
        "sdxl-base": {
            "name": "Stable Diffusion XL Base",
            "pipeline_class": StableDiffusionXLPipeline,
            "local_path": "sdxl/sdxl-base",
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
            "local_path": "stable-diffusion/sd-1.5",
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
            "local_path": "stable-diffusion/sd-2.1",
            "default_resolution": (768, 768),
            "vram_requirement": 6,
            "strengths": ["improved-quality", "composition", "versatile"],
            "use_cases": ["general purpose", "improved quality", "better composition"],
        },
    }

    AVAILABLE_MODELS = {
        "sdxl-base": {
            "name": "Stable Diffusion XL Base",
            "path": "stabilityai/stable-diffusion-xl-base-1.0",
            "local_path": "sdxl/sdxl-base",
            "type": "sdxl",
            "capabilities": ["txt2img", "img2img", "inpaint"],
            "strengths": ["photoreal", "high-quality", "commercial", "advertisement"],
            "recommended_for": ["photography", "marketing", "professional"],
            "vram_requirement": "12GB+",
            "supports_inpaint": True,
            "optimal_resolution": (1024, 1024),
            "max_resolution": (2048, 2048),
        },
        "sd-1.5": {
            "name": "Stable Diffusion 1.5",
            "path": "runwayml/stable-diffusion-v1-5",
            "local_path": "stable-diffusion/sd-1.5",
            "type": "sd",
            "capabilities": ["txt2img", "img2img", "inpaint"],
            "strengths": ["anime", "character", "lora-compatible", "fast"],
            "recommended_for": ["anime", "characters", "artistic", "creative"],
            "vram_requirement": "6GB+",
            "supports_inpaint": True,
            "optimal_resolution": (512, 512),
            "max_resolution": (1024, 1024),
        },
        "sd-2.1": {
            "name": "Stable Diffusion 2.1",
            "path": "stabilityai/stable-diffusion-2-1",
            "local_path": "stable-diffusion/sd-2.1",
            "type": "sd2",
            "capabilities": ["txt2img", "img2img", "inpaint"],
            "strengths": ["balanced", "versatile", "improved-quality"],
            "recommended_for": ["general", "mixed", "experimental"],
            "vram_requirement": "8GB+",
            "supports_inpaint": True,
            "optimal_resolution": (768, 768),
            "max_resolution": (1024, 1024),
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

    @classmethod
    def auto_select_model(cls, prompt: str, task_type: str = "txt2img") -> str:
        """Auto-select best model based on prompt analysis and task type"""
        prompt_lower = prompt.lower()

        # Keywords for different model preferences
        anime_keywords = [
            "anime",
            "manga",
            "character",
            "waifu",
            "kawaii",
            "chibi",
            "2d",
        ]
        photoreal_keywords = [
            "photo",
            "realistic",
            "portrait",
            "photography",
            "commercial",
            "professional",
        ]
        artistic_keywords = ["art", "painting", "artistic", "creative", "illustration"]

        # Count keyword matches
        anime_score = sum(1 for keyword in anime_keywords if keyword in prompt_lower)
        photoreal_score = sum(
            1 for keyword in photoreal_keywords if keyword in prompt_lower
        )
        artistic_score = sum(
            1 for keyword in artistic_keywords if keyword in prompt_lower
        )

        # Special considerations for inpainting
        if task_type == "inpaint":
            # SDXL generally better for inpainting
            if photoreal_score > 0 or len(prompt) > 100:
                return "sdxl-base"
            elif anime_score > photoreal_score:
                return "sd-1.5"
            else:
                return "sdxl-base"

        # General selection logic
        if anime_score > photoreal_score and anime_score > artistic_score:
            return "sd-1.5"  # Best for anime/character
        elif photoreal_score > 0 or len(prompt) > 150:
            return "sdxl-base"  # Best for detailed/photoreal
        elif artistic_score > 0:
            return "sd-2.1"  # Balanced for artistic
        else:
            return settings.PRIMARY_MODEL  # Default fallback

    @classmethod
    def get_optimal_dimensions(cls, model_id: str) -> tuple:
        """Get optimal dimensions for model"""
        model_info = cls.get_model_info(model_id)
        if model_info:
            return model_info.get("optimal_resolution", (512, 512))
        return (512, 512)


class SDModelManager:
    """Manage Stable Diffusion models with lazy loading"""

    def __init__(self):
        self._pipelines: Dict[str, Any] = {}
        self._current_model: Optional[str] = None

    def _get_model_path(self, model_name: str) -> str:
        """Get model path from shared cache"""
        return shared_cache.get_model_path(f"stable-diffusion/{model_name}")

    def list_available_models(self) -> List[str]:
        """List available SD models in cache"""
        # This would scan the cache directory
        # For now, return configured models
        return (
            [settings.SD_MODEL]
            if hasattr(settings, "SD_MODEL")
            else ["runwayml/stable-diffusion-v1-5"]
        )

    def load_model(self, model_name: str, **kwargs) -> Any:
        """Load SD model with optimized settings"""
        cache_key = f"sd:{model_name}"

        if cache_key in self._pipelines:
            logger.info(f"SD model {model_name} already loaded")
            return self._pipelines[cache_key]

        try:
            model_path = self._get_model_path(model_name)
            logger.info(f"Loading SD model: {model_name} from {model_path}")

            # Load pipeline with optimized settings
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=model_cache.dtype,
                cache_dir=model_path,
                local_files_only=settings.OFFLINE_MODE,
                safety_checker=None,  # Disable for performance
                requires_safety_checker=False,
                **kwargs,
            )

            # Move to device
            pipe = pipe.to(model_cache.device)

            # Optimize scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )

            # Apply memory optimizations
            if settings.ENABLE_ATTENTION_SLICING:
                pipe.enable_attention_slicing()
            if settings.ENABLE_XFORMERS and model_cache.device.type == "cuda":
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"XFormers not available: {e}")
            if settings.ENABLE_VAE_SLICING:
                pipe.enable_vae_slicing()
            if settings.ENABLE_CPU_OFFLOAD and model_cache.device.type == "cuda":
                pipe.enable_sequential_cpu_offload()

            self._pipelines[cache_key] = pipe
            self._current_model = model_name
            logger.info(f"SD model {model_name} loaded successfully")

            return pipe

        except Exception as e:
            logger.error(f"Failed to load SD model {model_name}: {e}")
            raise

    def get_model(self, model_name: Optional[str] = None) -> Any:
        """Get loaded model or load default"""
        if model_name is None:
            model_name = getattr(settings, "SD_MODEL", "runwayml/stable-diffusion-v1-5")

        cache_key = f"sd:{model_name}"
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]
        else:
            return self.load_model(model_name)  # type: ignore

    def unload_model(self, model_name: str):
        """Unload model to free memory"""
        cache_key = f"sd:{model_name}"
        if cache_key in self._pipelines:
            del self._pipelines[cache_key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"SD model {model_name} unloaded")

    def unload_all(self):
        """Unload all models"""
        for model_name in list(self._pipelines.keys()):
            self.unload_model(model_name.split(":")[1])


class ModelManager:
    """Enhanced model manager with multi-pipeline support"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Pipeline storage
        self.current_pipeline: Optional[
            Union[StableDiffusionPipeline, StableDiffusionXLPipeline]
        ] = None
        self.current_img2img_pipeline: Optional[
            Union[StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline]
        ] = None
        self.current_inpaint_pipeline: Optional[
            Union[StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline]
        ] = None

        # State tracking
        self.current_model_id: Optional[str] = None
        self.current_model: Optional[str] = None
        self.current_model_path: Optional[str] = None
        self.is_initialized: bool = False
        self.device: str = settings.DEVICE
        self.torch_dtype = settings.get_torch_dtype()
        self.startup_time: float = 0.0
        self.last_optimization_info: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}  # For future multi-model caching
        self.base_models_path = Path(settings.OUTPUT_PATH).parent / "models"

        # Performance tracking
        self.model_load_times: Dict[str, float] = {}
        self.generation_stats: Dict[str, Any] = {
            "total_generations": 0,
            "avg_generation_time": 0,
            "model_switches": 0,
        }

        self._initialized = True

    @property
    def available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available models from registry"""
        return ModelRegistry.AVAILABLE_MODELS

    def auto_select_model(self, prompt: str, task_type: str = "txt2img") -> str:
        """Auto-select model using registry logic"""
        return ModelRegistry.auto_select_model(prompt, task_type)

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
            if success:
                self.is_initialized = True
                logger.info("✅ ModelManager initialized successfully")
            else:
                logger.error("❌ ModelManager initialization failed")

            return success

        except Exception as e:
            logger.error(f"❌ ModelManager initialization failed: {str(e)}")
            return False

    async def _load_model(self, model_id: str) -> bool:
        """Load specific model with all pipeline variants"""
        try:
            model_info = ModelRegistry.get_model_info(model_id)
            if not model_info:
                raise ValueError(f"Unknown model: {model_id}")

            start_time = time.time()
            model_path = model_info["path"]
            model_type = model_info["type"]

            logger.info(f"🔄 Loading model: {model_id} ({model_type})")

            # Unload current model first
            await self._unload_current_model()

            # Load pipelines based on model type
            if model_type == "sdxl":
                await self._load_sdxl_pipelines(model_path)
            else:
                await self._load_sd_pipelines(model_path)

            # Apply optimizations
            self._apply_optimizations()

            # Update state
            self.current_model = model_id
            self.current_model_path = model_path
            load_time = time.time() - start_time
            self.model_load_times[model_id] = load_time

            logger.info(f"✅ Model loaded successfully: {model_id} ({load_time:.2f}s)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {str(e)}")
            await self._unload_current_model()
            return False

    async def _load_sdxl_pipelines(self, model_path: str):
        """Load SDXL pipeline variants"""
        # Text-to-image pipeline
        self.current_pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # Image-to-image pipeline (shared components)
        self.current_img2img_pipeline = StableDiffusionXLImg2ImgPipeline(
            vae=self.current_pipeline.vae,
            text_encoder=self.current_pipeline.text_encoder,
            text_encoder_2=self.current_pipeline.text_encoder_2,
            tokenizer=self.current_pipeline.tokenizer,
            tokenizer_2=self.current_pipeline.tokenizer_2,
            unet=self.current_pipeline.unet,
            scheduler=self.current_pipeline.scheduler,
        )

        # Inpainting pipeline (shared components)
        self.current_inpaint_pipeline = StableDiffusionXLInpaintPipeline(
            vae=self.current_pipeline.vae,
            text_encoder=self.current_pipeline.text_encoder,
            text_encoder_2=self.current_pipeline.text_encoder_2,
            tokenizer=self.current_pipeline.tokenizer,
            tokenizer_2=self.current_pipeline.tokenizer_2,
            unet=self.current_pipeline.unet,
            scheduler=self.current_pipeline.scheduler,
        )

    async def _load_sd_pipelines(self, model_path: str):
        """Load SD 1.5/2.1 pipeline variants"""
        # Text-to-image pipeline
        self.current_pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # Image-to-image pipeline (shared components)
        self.current_img2img_pipeline = StableDiffusionImg2ImgPipeline(
            feature_extractor=self.current_pipeline.feature_extractor,  # type: ignore
            vae=self.current_pipeline.vae,
            text_encoder=self.current_pipeline.text_encoder,
            tokenizer=self.current_pipeline.tokenizer,
            unet=self.current_pipeline.unet,
            scheduler=self.current_pipeline.scheduler,
            safety_checker=None,  # type: ignore
            requires_safety_checker=False,
        )

    def _apply_optimizations(self):
        """Apply performance optimizations to all pipelines"""
        pipelines = [
            self.current_pipeline,
            self.current_img2img_pipeline,
            self.current_inpaint_pipeline,
        ]

        for pipeline in pipelines:
            if pipeline is None:
                continue

            # Apply attention optimization
            setup_attention_processor(pipeline, force_sdpa=settings.USE_SDPA)

            # Memory optimizations
            if settings.ENABLE_CPU_OFFLOAD:
                pipeline.enable_sequential_cpu_offload()

            if settings.USE_ATTENTION_SLICING:
                pipeline.enable_attention_slicing()

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
        width: int = None,  # type: ignore
        height: int = None,  # type: ignore
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = None,  # type: ignore
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
        if not self.current_pipeline:
            raise RuntimeError("No model loaded")

        gen_logger = get_generation_logger("txt2img", self.current_model)  # type: ignore

        try:
            # Use model's optimal dimensions if not specified
            if width is None or height is None:
                opt_width, opt_height = ModelRegistry.get_optimal_dimensions(
                    self.current_model
                )
                width = width or opt_width
                height = height or opt_height

            # Prepare generation parameters
            generation_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
            }

            # Add seed if specified
            if seed is not None and seed != -1:
                generation_params["generator"] = torch.Generator(
                    device=self.device
                ).manual_seed(seed)

            gen_logger.info(f"Starting txt2img generation: {width}x{height}")

            # Generate
            with torch.inference_mode():
                result = self.current_pipeline(**generation_params)

            # Extract images safely
            images = self._extract_images_from_result(result)

            gen_logger.info(f"✅ Generated {len(images)} images")

            self.generation_stats["total_generations"] += len(images)

            return {
                "images": images,
                "generation_params": generation_params,
                "model_used": self.current_model,
            }

        except Exception as e:
            gen_logger.error(f"❌ txt2img generation failed: {str(e)}")
            raise

    async def generate_img2img(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str = "",
        strength: float = 0.75,
        width: int = None,  # type: ignore
        height: int = None,  # type: ignore
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = None,  # type: ignore
        num_images: int = 1,
    ) -> Dict[str, Any]:
        """Generate images using image-to-image pipeline"""
        if not self.current_img2img_pipeline:
            raise RuntimeError("No img2img pipeline loaded")

        gen_logger = get_generation_logger("img2img", self.current_model)  # type: ignore

        try:
            # Prepare generation parameters
            generation_params = {
                "prompt": prompt,
                "image": image,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
            }

            # Add dimensions if specified (for SDXL)
            if width and height:
                generation_params.update({"width": width, "height": height})

            # Add seed if specified
            if seed is not None and seed != -1:
                generation_params["generator"] = torch.Generator(
                    device=self.device
                ).manual_seed(seed)

            gen_logger.info(f"Starting img2img generation: strength={strength}")

            # Generate
            with torch.inference_mode():
                result = self.current_img2img_pipeline(**generation_params)

            # Extract images safely
            images = self._extract_images_from_result(result)

            gen_logger.info(f"✅ Generated {len(images)} images via img2img")

            self.generation_stats["total_generations"] += len(images)

            return {
                "images": images,
                "generation_params": generation_params,
                "model_used": self.current_model,
            }

        except Exception as e:
            gen_logger.error(f"❌ img2img generation failed: {str(e)}")
            raise

    async def generate_inpaint(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: str = "",
        strength: float = 0.75,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> Dict[str, Any]:
        """Generate images using inpainting pipeline"""
        if not self.current_inpaint_pipeline:
            raise RuntimeError("No inpaint pipeline loaded")

        gen_logger = get_generation_logger("inpaint", self.current_model)  # type: ignore

        try:
            # Prepare generation parameters
            generation_params = {
                "prompt": prompt,
                "image": image,
                "mask_image": mask_image,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
            }

            # Add dimensions if specified (for SDXL)
            if width and height:
                generation_params.update({"width": width, "height": height})

            # Add seed if specified
            if seed is not None and seed != -1:
                generation_params["generator"] = torch.Generator(
                    device=self.device
                ).manual_seed(seed)

            gen_logger.info(f"Starting inpaint generation: strength={strength}")

            # Generate
            with torch.inference_mode():
                result = self.current_inpaint_pipeline(**generation_params)

            # Extract images safely
            images = self._extract_images_from_result(result)

            gen_logger.info(f"✅ Generated {len(images)} images via inpaint")

            self.generation_stats["total_generations"] += len(images)

            return {
                "images": images,
                "generation_params": generation_params,
                "model_used": self.current_model,
            }

        except Exception as e:
            gen_logger.error(f"❌ inpaint generation failed: {str(e)}")
            raise

    def _extract_images_from_result(self, result: Any) -> List[Image.Image]:
        """Safely extract images from pipeline result"""
        try:
            # Handle different result types
            if hasattr(result, "images") and result.images:
                images = result.images
            elif isinstance(result, list):
                images = result
            elif isinstance(result, Image.Image):
                images = [result]
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")

            # Validate all items are PIL Images
            validated_images = []
            for i, img in enumerate(images):
                if isinstance(img, Image.Image):
                    validated_images.append(img)
                else:
                    logger.warning(f"Skipping non-PIL image at index {i}: {type(img)}")

            if not validated_images:
                raise ValueError("No valid PIL Images found in result")

            return validated_images

        except Exception as e:
            logger.error(f"❌ Failed to extract images: {str(e)}")
            raise

    def get_vram_usage(self) -> str:
        """Get current VRAM usage"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                return f"{allocated:.2f}GB / {reserved:.2f}GB"
            return "N/A (CPU mode)"
        except Exception:
            return "Unknown"

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and statistics"""
        return {
            "current_model": self.current_model,
            "current_model_path": self.current_model_path,
            "is_initialized": self.is_initialized,
            "device": self.device,
            "vram_usage": self.get_vram_usage(),
            "capabilities": {
                "txt2img": self.current_pipeline is not None,
                "img2img": self.current_img2img_pipeline is not None,
                "inpaint": self.current_inpaint_pipeline is not None,
            },
            "load_times": self.model_load_times,
            "generation_stats": self.generation_stats,
            "available_models": list(self.available_models.keys()),
        }

    async def _unload_current_model(self):
        """Unload current model to free memory"""
        try:
            pipelines = [
                ("txt2img", self.current_pipeline),
                ("img2img", self.current_img2img_pipeline),
                ("inpaint", self.current_inpaint_pipeline),
            ]

            for name, pipeline in pipelines:
                if pipeline is not None:
                    pipeline = pipeline.to("cpu")
                    del pipeline
                    logger.info(f"🗑️ Unloaded {name} pipeline")

            # Clear references
            self.current_pipeline = None
            self.current_img2img_pipeline = None
            self.current_inpaint_pipeline = None

            # Force memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("✅ Model unloaded and memory cleaned")

        except Exception as e:
            logger.error(f"❌ Error during model cleanup: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up resources and unload models."""
        logger.info("Cleaning up ModelManager...")

        if self.current_pipeline is not None:
            await self._unload_current_model()

        self.model_cache.clear()
        self.current_model = None
        self.current_model_path = None
        self.is_initialized = False
        logger.info("🧹 ModelManager cleanup completed")


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance (singleton pattern)."""
    global _model_manager

    if _model_manager is None:
        _model_manager = ModelManager()

    return _model_manager
