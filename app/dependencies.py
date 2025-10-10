"""
Dependency Injection providers for all services
"""

from functools import lru_cache
from typing import Generator
from fastapi import Depends

from .config import settings
from .shared_cache import shared_cache

# Import all services
from services.generation.txt2img_service import Txt2ImgService
from services.generation.img2img_service import Img2ImgService
from services.generation.inpaint_service import InpaintService
from services.postprocess.upscale_service import UpscaleService
from services.postprocess.face_restore_service import FaceRestoreService
from services.postprocess.video_service import VideoService
from services.processors.controlnet_service import ControlNetService
from services.models.caption_models import CaptionService
from services.models.vqa_models import VQAService
from services.queue.task_manager import TaskManager
from services.assets.asset_manager import AssetManager
from services.models.sd_models import SDModelManager
from services.models.vae_models import VAEModelManager
from services.models.controlnet_models import ControlNetModelManager
from services.models.lora_models import LoRAModelManager


@lru_cache()
def get_shared_cache():
    """Get shared cache instance"""
    return shared_cache


@lru_cache()
def get_task_manager() -> TaskManager:
    """Get task manager singleton"""
    return TaskManager()


@lru_cache()
def get_asset_manager() -> AssetManager:
    """Get asset manager singleton"""
    return AssetManager()


@lru_cache()
def get_txt2img_service() -> Txt2ImgService:
    """Get txt2img service with lazy initialization"""
    return Txt2ImgService()


@lru_cache()
def get_img2img_service() -> Img2ImgService:
    """Get img2img service with lazy initialization"""
    return Img2ImgService()


@lru_cache()
def get_inpaint_service() -> InpaintService:
    """Get inpainting service with lazy initialization"""
    return InpaintService()


@lru_cache()
def get_upscale_service() -> UpscaleService:
    """Get upscale service with lazy initialization"""
    return UpscaleService()


@lru_cache()
def get_face_restore_service() -> FaceRestoreService:
    """Get face restoration service with lazy initialization"""
    return FaceRestoreService()


@lru_cache()
def get_video_service() -> VideoService:
    """Get video service with lazy initialization"""
    return VideoService()


@lru_cache()
def get_controlnet_service() -> ControlNetService:
    """Get ControlNet service with lazy initialization"""
    return ControlNetService()


@lru_cache()
def get_caption_service() -> CaptionService:
    """Get caption service with lazy initialization"""
    return CaptionService()


@lru_cache()
def get_vqa_service() -> VQAService:
    """Get VQA service with lazy initialization"""
    return VQAService()


# Dependency providers for FastAPI
def txt2img_service() -> Generator[Txt2ImgService, None, None]:
    service = get_txt2img_service()
    try:
        yield service
    finally:
        # Cleanup if needed
        pass


def img2img_service() -> Generator[Img2ImgService, None, None]:
    service = get_img2img_service()
    try:
        yield service
    finally:
        pass


def inpainting_service() -> Generator[InpaintService, None, None]:
    service = get_inpaint_service()
    try:
        yield service
    finally:
        pass


def upscale_service() -> Generator[UpscaleService, None, None]:
    service = get_upscale_service()
    try:
        yield service
    finally:
        pass


def face_restore_service() -> Generator[FaceRestoreService, None, None]:
    service = get_face_restore_service()
    try:
        yield service
    finally:
        pass


def video_service() -> Generator[VideoService, None, None]:
    service = get_video_service()
    try:
        yield service
    finally:
        pass


def controlnet_service() -> Generator[ControlNetService, None, None]:
    service = get_controlnet_service()
    try:
        yield service
    finally:
        pass


def caption_service() -> Generator[CaptionService, None, None]:
    service = get_caption_service()
    try:
        yield service
    finally:
        pass


def vqa_service() -> Generator[VQAService, None, None]:
    service = get_vqa_service()
    try:
        yield service
    finally:
        pass


# Model Managers
@lru_cache()
def get_sd_model_manager() -> SDModelManager:
    return SDModelManager()


@lru_cache()
def get_vae_model_manager() -> VAEModelManager:
    return VAEModelManager()


@lru_cache()
def get_controlnet_model_manager() -> ControlNetModelManager:
    return ControlNetModelManager()


@lru_cache()
def get_lora_model_manager() -> LoRAModelManager:
    return LoRAModelManager()


# FastAPI Dependency Providers
def txt2img_service() -> Generator[Txt2ImgService, None, None]:
    service = get_txt2img_service()
    try:
        yield service
    finally:
        pass


def img2img_service() -> Generator[Img2ImgService, None, None]:
    service = get_img2img_service()
    try:
        yield service
    finally:
        pass


def inpainting_service() -> Generator[InpaintService, None, None]:
    service = get_inpaint_service()
    try:
        yield service
    finally:
        pass


def upscale_service() -> Generator[UpscaleService, None, None]:
    service = get_upscale_service()
    try:
        yield service
    finally:
        pass


def face_restore_service() -> Generator[FaceRestoreService, None, None]:
    service = get_face_restore_service()
    try:
        yield service
    finally:
        pass


def video_service() -> Generator[VideoService, None, None]:
    service = get_video_service()
    try:
        yield service
    finally:
        pass


def controlnet_service() -> Generator[ControlNetService, None, None]:
    service = get_controlnet_service()
    try:
        yield service
    finally:
        pass


def caption_service() -> Generator[CaptionService, None, None]:
    service = get_caption_service()
    try:
        yield service
    finally:
        pass


def vqa_service() -> Generator[VQAService, None, None]:
    service = get_vqa_service()
    try:
        yield service
    finally:
        pass


def task_manager() -> Generator[TaskManager, None, None]:
    manager = get_task_manager()
    try:
        yield manager
    finally:
        pass


def asset_manager() -> Generator[AssetManager, None, None]:
    manager = get_asset_manager()
    try:
        yield manager
    finally:
        pass


def sd_model_manager() -> Generator[SDModelManager, None, None]:
    manager = get_sd_model_manager()
    try:
        yield manager
    finally:
        pass


def vae_model_manager() -> Generator[VAEModelManager, None, None]:
    manager = get_vae_model_manager()
    try:
        yield manager
    finally:
        pass


def controlnet_model_manager() -> Generator[ControlNetModelManager, None, None]:
    manager = get_controlnet_model_manager()
    try:
        yield manager
    finally:
        pass


def lora_model_manager() -> Generator[LoRAModelManager, None, None]:
    manager = get_lora_model_manager()
    try:
        yield manager
    finally:
        pass
