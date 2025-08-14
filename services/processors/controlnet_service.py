# services/processors/controlnet_service.py
"""
ControlNet processor management with optimized pipeline handling
"""
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union, List
import logging
from pathlib import Path
import base64
from io import BytesIO

from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionControlNetPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import (
    StableDiffusionXLControlNetPipeline,
)


from controlnet_aux import (
    CannyDetector,
    OpenposeDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
)

from app.config import settings
from utils.logging_utils import get_generation_logger
from utils.attention_utils import setup_attention_processor
from utils.image_utils import preprocess_controlnet_image

logger = logging.getLogger(__name__)


class ControlNetProcessor:
    """Individual ControlNet processor with caching"""

    def __init__(self, controlnet_type: str, model_path: str):
        self.controlnet_type = controlnet_type
        self.model_path = model_path
        self.controlnet_model: Optional[ControlNetModel] = None
        self.preprocessor = None
        self.is_loaded = False

    async def load(self) -> bool:
        """Load ControlNet model and preprocessor"""
        try:
            logger.info(f"Loading ControlNet: {self.controlnet_type}")

            # Load ControlNet model
            self.controlnet_model = ControlNetModel.from_pretrained(
                self.model_path,
                torch_dtype=settings.get_torch_dtype(),
                device_map="auto" if settings.DEVICE == "cuda" else None,
            ).to(
                settings.DEVICE  # type: ignore
            )

            # Load corresponding preprocessor
            self.preprocessor = self._load_preprocessor()

            self.is_loaded = True
            logger.info(f"✅ ControlNet {self.controlnet_type} loaded successfully")
            return True

        except Exception as e:
            logger.error(
                f"❌ Failed to load ControlNet {self.controlnet_type}: {str(e)}"
            )
            return False

    def _load_preprocessor(self):
        """Load the appropriate preprocessor for this ControlNet type"""
        preprocessors = {
            "canny": lambda: CannyDetector(),
            "openpose": lambda: OpenposeDetector.from_pretrained(
                "lllyasviel/Annotators"
            ),
            "depth": lambda: MidasDetector.from_pretrained("lllyasviel/Annotators"),
            "mlsd": lambda: MLSDdetector.from_pretrained("lllyasviel/Annotators"),
            "normal": lambda: NormalBaeDetector.from_pretrained(
                "lllyasviel/Annotators"
            ),
            "scribble": lambda: None,  # Manual scribble, no auto-preprocessing
        }

        preprocessor_loader = preprocessors.get(self.controlnet_type)
        if preprocessor_loader:
            return preprocessor_loader()
        return None

    def preprocess_image(self, image: Image.Image, **kwargs) -> Image.Image:
        """Process input image to generate control condition"""
        if not self.preprocessor:
            # For manual conditions like scribble, return as-is
            return image

        try:
            if self.controlnet_type == "canny":
                # Canny edge detection with configurable thresholds
                low_threshold = kwargs.get("canny_low", 50)
                high_threshold = kwargs.get("canny_high", 200)
                return self.preprocessor(image, low_threshold, high_threshold)

            elif self.controlnet_type == "openpose":
                # Human pose estimation
                return self.preprocessor(image)

            elif self.controlnet_type == "depth":
                # Depth estimation
                return self.preprocessor(image)

            elif self.controlnet_type == "mlsd":
                # Line segment detection
                return self.preprocessor(image)

            elif self.controlnet_type == "normal":
                # Surface normal estimation
                return self.preprocessor(image)

            else:
                logger.warning(f"Unknown ControlNet type: {self.controlnet_type}")
                return image

        except Exception as e:
            logger.error(f"Preprocessing failed for {self.controlnet_type}: {str(e)}")
            return image

    def unload(self):
        """Unload model to free memory"""
        if self.controlnet_model:
            self.controlnet_model = self.controlnet_model.to("cpu")  # type: ignore
            del self.controlnet_model
            self.controlnet_model = None

        if self.preprocessor:
            del self.preprocessor
            self.preprocessor = None

        self.is_loaded = False
        torch.cuda.empty_cache()


class ControlNetManager:
    """Centralized ControlNet management with model caching"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.processors: Dict[str, ControlNetProcessor] = {}
        self.supported_types = [
            "canny",
            "openpose",
            "depth",
            "mlsd",
            "normal",
            "scribble",
        ]
        self.current_pipeline: Optional[
            Union[
                StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
            ]
        ] = None
        self._initialized = True

    async def initialize(
        self, controlnet_types: List[str] = ["canny", "openpose"]
    ) -> bool:
        """Initialize specified ControlNet processors"""
        if controlnet_types is None:
            controlnet_types = ["canny", "openpose"]  # Default minimal set

        logger.info(f"Initializing ControlNet processors: {controlnet_types}")

        success_count = 0
        for controlnet_type in controlnet_types:
            if controlnet_type not in self.supported_types:
                logger.warning(f"Unsupported ControlNet type: {controlnet_type}")
                continue

            processor = ControlNetProcessor(
                controlnet_type=controlnet_type,
                model_path=self._get_model_path(controlnet_type),
            )

            if await processor.load():
                self.processors[controlnet_type] = processor
                success_count += 1
            else:
                logger.error(f"Failed to load {controlnet_type}")

        logger.info(
            f"✅ Loaded {success_count}/{len(controlnet_types)} ControlNet processors"
        )
        return success_count > 0

    def _get_model_path(self, controlnet_type: str) -> str:
        """Get model path for specific ControlNet type"""
        base_path = Path(settings.CONTROLNET_PATH)

        # Standard model mappings
        model_mappings = {
            "canny": "sd-controlnet-canny",
            "openpose": "sd-controlnet-openpose",
            "depth": "sd-controlnet-depth",
            "mlsd": "sd-controlnet-mlsd",
            "normal": "sd-controlnet-normal",
            "scribble": "sd-controlnet-scribble",
        }

        model_name = model_mappings.get(
            controlnet_type, f"sd-controlnet-{controlnet_type}"
        )
        return str(base_path / model_name)

    async def create_pipeline(self, base_model_path: str, controlnet_type: str) -> bool:
        """Create ControlNet pipeline with base model"""
        try:
            if controlnet_type not in self.processors:
                logger.error(f"ControlNet {controlnet_type} not loaded")
                return False

            processor = self.processors[controlnet_type]

            # Determine pipeline type based on model
            if "xl" in base_model_path.lower():
                pipeline_class = StableDiffusionXLControlNetPipeline
            else:
                pipeline_class = StableDiffusionControlNetPipeline

            # Create pipeline
            self.current_pipeline = pipeline_class.from_pretrained(
                base_model_path,
                controlnet=processor.controlnet_model,
                torch_dtype=settings.get_torch_dtype(),
                device_map="auto" if settings.DEVICE == "cuda" else None,
                safety_checker=None,  # Disable for performance
                requires_safety_checker=False,
            ).to(settings.DEVICE)

            # Apply optimizations
            setup_attention_processor(
                self.current_pipeline, force_sdpa=settings.USE_SDPA
            )

            if settings.ENABLE_CPU_OFFLOAD:
                self.current_pipeline.enable_sequential_cpu_offload()

            if settings.USE_ATTENTION_SLICING:
                self.current_pipeline.enable_attention_slicing()

            logger.info(f"✅ ControlNet pipeline created: {controlnet_type}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create ControlNet pipeline: {str(e)}")
            return False

    async def generate_with_controlnet(
        self,
        prompt: str,
        control_image: Image.Image,
        controlnet_type: str,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """Generate image with ControlNet conditioning"""
        gen_logger = get_generation_logger("controlnet", controlnet_type)

        try:
            if not self.current_pipeline:
                raise ValueError("No ControlNet pipeline loaded")

            # Preprocess control image
            processor = self.processors[controlnet_type]
            processed_control = processor.preprocess_image(
                control_image, **generation_kwargs.get("controlnet_params", {})
            )

            # Generation parameters
            generation_params = {
                "prompt": prompt,
                "image": processed_control,
                "num_inference_steps": generation_kwargs.get("num_inference_steps", 25),
                "guidance_scale": generation_kwargs.get("guidance_scale", 7.5),
                "controlnet_conditioning_scale": generation_kwargs.get(
                    "controlnet_strength", 1.0
                ),
                "generator": (
                    torch.Generator(device=settings.DEVICE).manual_seed(
                        generation_kwargs.get("seed", -1)
                    )
                    if generation_kwargs.get("seed", -1) != -1
                    else None
                ),
            }

            # Generate
            gen_logger.info(f"Starting ControlNet generation: {controlnet_type}")
            with torch.inference_mode():
                result = self.current_pipeline(**generation_params)

            # Extract images safely
            if hasattr(result, "images") and result.images:  # type: ignore
                images = result.images  # type: ignore
            elif isinstance(result, list):
                images = result
            else:
                raise ValueError("Unexpected pipeline output format")

            gen_logger.info(f"✅ ControlNet generation completed: {len(images)} images")

            return {
                "images": images,
                "controlnet_type": controlnet_type,
                "control_image": processed_control,
                "generation_params": generation_params,
            }

        except Exception as e:
            gen_logger.error(f"❌ ControlNet generation failed: {str(e)}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get current ControlNet status"""
        return {
            "loaded_processors": list(self.processors.keys()),
            "supported_types": self.supported_types,
            "pipeline_loaded": self.current_pipeline is not None,
            "total_vram_usage": self._estimate_vram_usage(),
        }

    def _estimate_vram_usage(self) -> str:
        """Estimate current VRAM usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            return f"{allocated:.2f}GB"
        return "N/A"

    async def cleanup(self):
        """Clean up all loaded models"""
        logger.info("Cleaning up ControlNet models...")

        for processor in self.processors.values():
            processor.unload()

        if self.current_pipeline:
            self.current_pipeline = self.current_pipeline.to("cpu")
            del self.current_pipeline
            self.current_pipeline = None

        self.processors.clear()
        torch.cuda.empty_cache()
        logger.info("✅ ControlNet cleanup completed")


# Global singleton instance
def get_controlnet_manager() -> ControlNetManager:
    """Get global ControlNet manager instance"""
    return ControlNetManager()
