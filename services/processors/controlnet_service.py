# services/processors/controlnet_service.py
"""
ControlNet processor management with optimized pipeline handling
"""
import torch
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Callable
import logging
from pathlib import Path

from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)


try:
    from controlnet_aux import (  # type: ignore
        CannyDetector,
        OpenposeDetector,
        MidasDetector,
        MLSDdetector,
        NormalBaeDetector,
    )

    CONTROLNET_AUX_AVAILABLE = True
except Exception:  # pragma: no cover
    CannyDetector = None  # type: ignore[assignment]
    OpenposeDetector = None  # type: ignore[assignment]
    MidasDetector = None  # type: ignore[assignment]
    MLSDdetector = None  # type: ignore[assignment]
    NormalBaeDetector = None  # type: ignore[assignment]
    CONTROLNET_AUX_AVAILABLE = False

from app.config import settings
from utils.logging_utils import get_generation_logger
from utils.attention_utils import setup_attention_processor

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

            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"ControlNet weights not found at: {self.model_path} (expected local path under {settings.CONTROLNET_PATH})"
                )

            # Load ControlNet model
            self.controlnet_model = ControlNetModel.from_pretrained(
                self.model_path,
                torch_dtype=settings.get_torch_dtype(),
                device_map="auto" if settings.DEVICE == "cuda" else None,
            ).to(
                settings.DEVICE  # type: ignore
            )

            # Load corresponding preprocessor (best-effort; ControlNet can still run
            # with user-supplied, already preprocessed condition images).
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
        if self.controlnet_type == "scribble":
            return None

        if not CONTROLNET_AUX_AVAILABLE:
            logger.warning(
                "ControlNet auto-preprocess unavailable (missing `controlnet-aux`). "
                "Set controlnet.preprocess=false and provide a preprocessed condition image."
            )
            return None

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
            try:
                return preprocessor_loader()
            except Exception as e:
                logger.warning(
                    "ControlNet preprocessor load failed; auto-preprocess disabled. "
                    "Set controlnet.preprocess=false or ensure annotators are available.",
                    extra={"controlnet_type": self.controlnet_type, "error": str(e)},
                )
                return None
        return None

    def preprocess_image(
        self, image: Image.Image, preprocess: bool = True, **kwargs
    ) -> Image.Image:
        """Process input image to generate control condition"""
        if not preprocess or self.controlnet_type == "scribble":
            return image

        if not self.preprocessor:
            raise RuntimeError(
                "ControlNet auto-preprocess is not available. "
                "Install `controlnet-aux` (and annotators), or set controlnet.preprocess=false "
                "and pass a preprocessed condition image."
            )

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
            raise RuntimeError(
                "ControlNet auto-preprocess failed. "
                "Set controlnet.preprocess=false to skip preprocessing, or fix annotator installation."
            ) from e

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

        # NOTE: processors are keyed by "<variant>:<type>" where variant is "sd" or "sdxl".
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
                StableDiffusionControlNetPipeline,
                StableDiffusionXLControlNetPipeline,
                StableDiffusionControlNetImg2ImgPipeline,
                StableDiffusionXLControlNetImg2ImgPipeline,
            ]
        ] = None
        self.current_pipeline_key: Optional[tuple[str, str, str]] = None
        self._initialized = True

    async def initialize(
        self, controlnet_types: List[str] = ["canny", "openpose"], variant: str = "sd"
    ) -> bool:
        """Initialize specified ControlNet processors.

        variant:
        - "sd":   SD 1.5/2.x ControlNet weights
        - "sdxl": SDXL ControlNet weights (only some types available)
        """
        if controlnet_types is None:
            controlnet_types = ["canny", "openpose"]  # Default minimal set

        logger.info(f"Initializing ControlNet processors: {controlnet_types}")

        success_count = 0
        for controlnet_type in controlnet_types:
            if controlnet_type not in self.supported_types:
                logger.warning(f"Unsupported ControlNet type: {controlnet_type}")
                continue
            processor_key = self._get_processor_key(controlnet_type, variant)
            existing = self.processors.get(processor_key)
            if existing and existing.is_loaded:
                success_count += 1
                continue

            processor = ControlNetProcessor(
                controlnet_type=controlnet_type,
                model_path=self._get_model_path(controlnet_type, variant),
            )

            if await processor.load():
                self.processors[processor_key] = processor
                success_count += 1
            else:
                logger.error(f"Failed to load {controlnet_type}")

        logger.info(
            f"✅ Loaded {success_count}/{len(controlnet_types)} ControlNet processors"
        )
        return success_count > 0

    def _get_processor_key(self, controlnet_type: str, variant: str) -> str:
        return f"{variant}:{controlnet_type}"

    def _infer_variant_from_base_model(self, base_model_path: str) -> str:
        return "sdxl" if "xl" in str(base_model_path).lower() else "sd"

    def _get_model_path(self, controlnet_type: str, variant: str) -> str:
        """Get local model path for specific ControlNet type/variant.

        Storage layout follows ~/Desktop/data_model_structure.md:
        - SD weights:   {CONTROLNET_PATH}/sd/<type>
        - SDXL weights: {CONTROLNET_PATH}/sdxl/<type>-sdxl (only canny/openpose/depth)
        """
        base_path = Path(settings.CONTROLNET_PATH)

        if variant == "sdxl":
            sdxl_map = {
                "canny": "canny-sdxl",
                "openpose": "openpose-sdxl",
                "depth": "depth-sdxl",
            }
            model_dir = sdxl_map.get(controlnet_type)
            if not model_dir:
                raise ValueError(
                    f"ControlNet type '{controlnet_type}' does not have an SDXL weight mapping"
                )
            return str((base_path / "sdxl" / model_dir).resolve())

        # Default: SD 1.5/2.x weights
        return str((base_path / "sd" / controlnet_type).resolve())

    async def create_pipeline(
        self, base_model_path: str, controlnet_type: str, pipeline_mode: str = "img2img"
    ) -> bool:
        """Create ControlNet pipeline with base model.

        pipeline_mode:
        - img2img: StableDiffusion{XL}ControlNetImg2ImgPipeline (uses init image + control image)
        - txt2img: StableDiffusion{XL}ControlNetPipeline (uses only control image)
        """
        try:
            if pipeline_mode not in {"img2img", "txt2img"}:
                raise ValueError(f"Unsupported pipeline_mode: {pipeline_mode}")

            variant = self._infer_variant_from_base_model(base_model_path)
            processor_key = self._get_processor_key(controlnet_type, variant)
            if processor_key not in self.processors:
                await self.initialize([controlnet_type], variant=variant)
            if processor_key not in self.processors:
                logger.error(
                    f"ControlNet {controlnet_type} not loaded",
                    extra={"variant": variant, "base_model_path": base_model_path},
                )
                return False

            pipeline_key = (base_model_path, controlnet_type, pipeline_mode)
            if self.current_pipeline is not None and self.current_pipeline_key == pipeline_key:
                return True

            processor = self.processors[processor_key]

            # Determine pipeline type based on model
            is_xl = "xl" in base_model_path.lower()
            if pipeline_mode == "img2img":
                pipeline_class = (
                    StableDiffusionXLControlNetImg2ImgPipeline
                    if is_xl
                    else StableDiffusionControlNetImg2ImgPipeline
                )
            else:
                pipeline_class = (
                    StableDiffusionXLControlNetPipeline
                    if is_xl
                    else StableDiffusionControlNetPipeline
                )

            # Create pipeline
            self.current_pipeline = pipeline_class.from_pretrained(
                base_model_path,
                controlnet=processor.controlnet_model,
                torch_dtype=settings.get_torch_dtype(),
                device_map="auto" if settings.DEVICE == "cuda" else None,
                safety_checker=None,  # Disable for performance
                requires_safety_checker=False,
            ).to(settings.DEVICE)
            self.current_pipeline_key = pipeline_key

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
        init_image: Optional[Image.Image],
        control_image: Image.Image,
        controlnet_type: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        callback_steps: int = 1,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """Generate image with ControlNet conditioning"""
        gen_logger = get_generation_logger("controlnet", controlnet_type)

        try:
            if not self.current_pipeline:
                raise ValueError("No ControlNet pipeline loaded")

            # Preprocess control image
            if not self.current_pipeline_key:
                raise ValueError("No ControlNet pipeline key found")
            variant = self._infer_variant_from_base_model(self.current_pipeline_key[0])
            processor_key = self._get_processor_key(controlnet_type, variant)
            if processor_key not in self.processors:
                raise ValueError(
                    f"ControlNet {controlnet_type} not loaded for variant={variant}"
                )
            processor = self.processors[processor_key]
            controlnet_params = generation_kwargs.get("controlnet_params", {}) or {}
            processed_control = processor.preprocess_image(
                control_image,
                preprocess=bool(controlnet_params.get("preprocess", True)),
                **controlnet_params,
            )

            seed = generation_kwargs.get("seed", None)
            generator = None
            if seed is not None and seed != -1:
                generator = torch.Generator(device=settings.DEVICE).manual_seed(int(seed))

            negative_prompt = generation_kwargs.get("negative_prompt", "")
            num_inference_steps = generation_kwargs.get("num_inference_steps", 25)
            guidance_scale = generation_kwargs.get("guidance_scale", 7.5)
            controlnet_strength = generation_kwargs.get("controlnet_strength", 1.0)
            strength = generation_kwargs.get("strength", 0.75)
            guidance_start = generation_kwargs.get("guidance_start", 0.0)
            guidance_end = generation_kwargs.get("guidance_end", 1.0)

            if self.current_pipeline_key and self.current_pipeline_key[2] == "txt2img":
                # txt2img: pipeline expects control condition as `image`.
                pipeline_params: Dict[str, Any] = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": processed_control,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "controlnet_conditioning_scale": controlnet_strength,
                    "generator": generator,
                }
            else:
                # img2img: pipeline expects init image + control image.
                if init_image is None:
                    raise ValueError("init_image is required for ControlNet img2img")
                pipeline_params = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": init_image,
                    "control_image": processed_control,
                    "strength": strength,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "controlnet_conditioning_scale": controlnet_strength,
                    "control_guidance_start": guidance_start,
                    "control_guidance_end": guidance_end,
                    "generator": generator,
                }

                width = generation_kwargs.get("width", None)
                height = generation_kwargs.get("height", None)
                if isinstance(width, int) and isinstance(height, int):
                    pipeline_params.update({"width": width, "height": height})

            if progress_callback is not None:
                stride = max(1, int(callback_steps))

                def _cb(step: int, timestep: int, latents):  # type: ignore[no-untyped-def]
                    try:
                        progress_callback(step, int(num_inference_steps))
                    except Exception:
                        pass

                pipeline_params["callback"] = _cb
                pipeline_params["callback_steps"] = stride

            # Generate
            gen_logger.info(f"Starting ControlNet generation: {controlnet_type}")
            with torch.inference_mode():
                result = self.current_pipeline(**pipeline_params)  # type: ignore[misc]

            # Extract images safely
            if hasattr(result, "images") and result.images:  # type: ignore
                images = result.images  # type: ignore
            elif isinstance(result, list):
                images = result
            else:
                raise ValueError("Unexpected pipeline output format")

            gen_logger.info(f"✅ ControlNet generation completed: {len(images)} images")

            init_size = (
                (init_image.width, init_image.height) if init_image is not None else None
            )
            return {
                "images": images,
                "controlnet_type": controlnet_type,
                "init_image_size": init_size,
                "control_image_size": (processed_control.width, processed_control.height),
                "generation_params": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "strength": strength,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "controlnet_conditioning_scale": controlnet_strength,
                    "seed": seed,
                    "control_guidance_start": guidance_start,
                    "control_guidance_end": guidance_end,
                    "pipeline_mode": (
                        self.current_pipeline_key[2] if self.current_pipeline_key else "img2img"
                    ),
                },
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
            "pipeline_mode": (self.current_pipeline_key[2] if self.current_pipeline_key else None),
            "total_vram_usage": self._estimate_vram_usage(),
            "preprocess_available": CONTROLNET_AUX_AVAILABLE,
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
