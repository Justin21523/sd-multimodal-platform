"""
Caption models management with lazy loading and device optimization
"""

import logging
from typing import Dict, Any, Optional
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from app.config import settings
from app.shared_cache import shared_cache
from .model_cache import model_cache

logger = logging.getLogger(__name__)


class CaptionService:
    """Caption generation service with lazy loading"""

    def __init__(self):
        self._model = None
        self._processor = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy load model with optimized settings"""
        if self._model_loaded:
            return

        try:
            model_name = settings.CAPTION_MODEL
            cache_dir = shared_cache.get_model_path(model_name)

            logger.info(f"Loading caption model: {model_name}")

            # Load processor and model with optimized settings
            self._processor = Blip2Processor.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=settings.OFFLINE_MODE
            )

            self._model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=settings.OFFLINE_MODE,
                torch_dtype=model_cache.dtype,
                device_map=(
                    model_cache.device.type
                    if model_cache.device.type != "cpu"
                    else None
                ),
            )

            # Apply memory optimizations
            model_cache.setup_memory_optimizations()

            self._model_loaded = True
            logger.info(f"Caption model loaded successfully on {model_cache.device}")

        except Exception as e:
            logger.error(f"Failed to load caption model: {e}")
            raise

    def generate_caption(self, image, max_length: int = 50) -> str:
        """Generate caption for image"""
        if not self._model_loaded:
            self._load_model()

        try:
            # Process image and generate caption
            inputs = self._processor(image, return_tensors="pt").to(model_cache.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_length=max_length, num_beams=5, early_stopping=True
                )

            caption = self._processor.decode(outputs[0], skip_special_tokens=True)
            return caption.strip()

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            raise

    def unload_model(self):
        """Unload model to free memory"""
        if self._model:
            del self._model
            self._model = None

        if self._processor:
            del self._processor
            self._processor = None

        self._model_loaded = False
        logger.info("Caption model unloaded")
