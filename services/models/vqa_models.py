"""
VQA / multimodal chat model service with lazy loading.

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


class VQAService:
    """Visual Question Answering service with lazy model initialization."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._model_loaded = False

    def _load_model(self) -> None:
        if self._model_loaded:
            return

        model_name = settings.VQA_MODEL
        logger.info("Loading VQA model: %s", model_name)

        try:
            if "llava" in model_name.lower():
                from transformers import LlavaForConditionalGeneration, LlavaProcessor

                self._processor = LlavaProcessor.from_pretrained(
                    model_name, local_files_only=settings.OFFLINE_MODE
                )
                self._model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    local_files_only=settings.OFFLINE_MODE,
                    torch_dtype=model_cache.dtype,
                    device_map="auto" if model_cache.device.type != "cpu" else None,
                )

            elif "qwen" in model_name.lower():
                from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

                self._processor = Qwen2VLProcessor.from_pretrained(
                    model_name, local_files_only=settings.OFFLINE_MODE
                )
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    local_files_only=settings.OFFLINE_MODE,
                    torch_dtype=model_cache.dtype,
                    device_map="auto" if model_cache.device.type != "cpu" else None,
                )

            else:
                raise ValueError(f"Unsupported VQA model: {model_name}")

            if model_cache.device.type == "cpu":
                self._model.to(model_cache.device)  # type: ignore[union-attr]

            model_cache.setup_memory_optimizations()
            self._model_loaded = True
            logger.info("VQA model loaded on %s", model_cache.device)

        except Exception as exc:
            logger.error("Failed to load VQA model: %s", exc)
            raise

    def answer_question(self, image: Any, question: str, **kwargs: Any) -> str:
        """Answer a question about an image."""
        if not self._model_loaded:
            self._load_model()

        if self._processor is None or self._model is None:
            raise RuntimeError("VQA model is not initialized")

        model_name = settings.VQA_MODEL.lower()

        if "llava" in model_name:
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        elif "qwen" in model_name and hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(  # type: ignore[union-attr]
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            inputs = self._processor(image, question, return_tensors="pt")

        if hasattr(inputs, "to"):
            inputs = inputs.to(model_cache.device)

        with torch.no_grad():
            if isinstance(inputs, dict):
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=int(kwargs.get("max_new_tokens", 100)),
                    do_sample=bool(kwargs.get("do_sample", True)),
                    temperature=float(kwargs.get("temperature", 0.7)),
                )
            else:
                outputs = self._model.generate(  # type: ignore[arg-type]
                    inputs,
                    max_new_tokens=int(kwargs.get("max_new_tokens", 100)),
                    do_sample=bool(kwargs.get("do_sample", True)),
                    temperature=float(kwargs.get("temperature", 0.7)),
                )

        answer = self._processor.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

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
