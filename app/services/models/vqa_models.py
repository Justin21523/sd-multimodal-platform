"""
VQA models management with lazy loading
"""

import logging
import torch
from typing import Dict, Any, Optional
from app.config import settings
from app.shared_cache import shared_cache
from .model_cache import model_cache

logger = logging.getLogger(__name__)


class VQAService:
    """Visual Question Answering service with lazy loading"""

    def __init__(self):
        self._model = None
        self._processor = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy load VQA model"""
        if self._model_loaded:
            return

        try:
            model_name = settings.VQA_MODEL
            cache_dir = shared_cache.get_model_path(model_name)

            logger.info(f"Loading VQA model: {model_name}")

            # Choose between LLaVA and Qwen-VL based on config
            if "llava" in model_name.lower():
                from transformers import LlavaForConditionalGeneration, LlavaProcessor

                self._processor = LlavaProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=settings.OFFLINE_MODE,
                )
                self._model = LlavaForConditionalGeneration.from_pretrained(
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
            elif "qwen" in model_name.lower():
                from transformers import (
                    Qwen2VLForConditionalGeneration,
                    Qwen2VLProcessor,
                )

                self._processor = Qwen2VLProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=settings.OFFLINE_MODE,
                )
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
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
            else:
                raise ValueError(f"Unsupported VQA model: {model_name}")

            model_cache.setup_memory_optimizations()
            self._model_loaded = True
            logger.info(f"VQA model loaded successfully on {model_cache.device}")

        except Exception as e:
            logger.error(f"Failed to load VQA model: {e}")
            raise

    def answer_question(self, image, question: str, **kwargs) -> str:
        """Answer question about image"""
        if not self._model_loaded:
            self._load_model()

        try:
            # Prepare conversation format based on model type
            if "llava" in settings.VQA_MODEL.lower():
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                inputs = self._processor(text=prompt, images=image, return_tensors="pt")
            elif "qwen" in settings.VQA_MODEL.lower():
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                inputs = self._processor.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
            else:
                inputs = self._processor(image, question, return_tensors="pt")

            inputs = inputs.to(model_cache.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 100),
                    do_sample=kwargs.get("do_sample", True),
                    temperature=kwargs.get("temperature", 0.7),
                )

            answer = self._processor.decode(outputs[0], skip_special_tokens=True)
            return answer.strip()

        except Exception as e:
            logger.error(f"VQA inference failed: {e}")
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
        logger.info("VQA model unloaded")
