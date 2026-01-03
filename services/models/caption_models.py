"""
Caption model service (BLIP-2) with lazy loading.

Canonical implementation lives under `services/` so both sync API and workers
can share the same logic. Storage must follow `~/Desktop/data_model_structure.md`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from app.config import settings
from app.shared_cache import shared_cache  # noqa: F401  (side-effect: set cache env vars)
from services.models.model_cache import model_cache

logger = logging.getLogger(__name__)


class CaptionService:
    """Image captioning service with lazy model initialization."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._model_loaded = False

    def _load_model(self) -> None:
        if self._model_loaded:
            return

        model_name = settings.CAPTION_MODEL
        logger.info("Loading caption model: %s", model_name)

        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor

            self._processor = Blip2Processor.from_pretrained(
                model_name, local_files_only=settings.OFFLINE_MODE
            )
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                local_files_only=settings.OFFLINE_MODE,
                torch_dtype=model_cache.dtype,
                device_map="auto" if model_cache.device.type != "cpu" else None,
            )
            if model_cache.device.type == "cpu":
                self._model.to(model_cache.device)

            model_cache.setup_memory_optimizations()
            self._model_loaded = True
            logger.info("Caption model loaded on %s", model_cache.device)

        except Exception as exc:
            logger.error("Failed to load caption model: %s", exc)
            raise

    def generate_caption(self, image: Any, *, max_length: int = 50) -> str:
        """Generate a caption for a PIL image."""
        if not self._model_loaded:
            self._load_model()

        if self._processor is None or self._model is None:
            raise RuntimeError("Caption model is not initialized")

        inputs = self._processor(image, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model_cache.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_length=max_length, num_beams=5, early_stopping=True
            )

        caption = self._processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._model_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
