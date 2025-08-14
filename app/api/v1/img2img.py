# app/api/v1/img2img.py
"""
Image-to-image generation API endpoints with ControlNet support
"""
from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import logging
import time
import asyncio
import base64
from io import BytesIO
from PIL import Image

from app.schemas.requests import Img2ImgRequest, InpaintRequest
from app.schemas.responses import GenerationResponse
from services.models.sd_models import get_model_manager
from services.processors.controlnet_service import get_controlnet_manager
from utils.logging_utils import get_request_logger, get_generation_logger
from utils.image_utils import (
    base64_to_pil_image,
    pil_image_to_base64,
    prepare_img2img_image,
    prepare_inpaint_mask,
)
from utils.file_utils import save_generation_output
from utils.metadata_utils import save_generation_metadata
from app.config import settings

router = APIRouter(prefix="/img2img", tags=["Image-to-Image Generation"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=GenerationResponse)
async def generate_img2img(
    request: Img2ImgRequest, background_tasks: BackgroundTasks, http_request: Request
) -> Dict[str, Any]:
    """
    Generate image variations from source image and text prompt

    Supports:
    - Strength control for noise injection level
    - Optional ControlNet conditioning
    - Model auto-selection based on prompt/style
    - Full metadata tracking and reproducibility
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    req_logger = get_request_logger(request_id)
    gen_logger = get_generation_logger(request_id, "img2img")

    start_time = time.time()

    try:
        req_logger.info(
            "🎨 Starting img2img generation",
            extra={
                "prompt_length": len(request.prompt),
                "has_controlnet": request.controlnet is not None,
                "strength": request.strength,
                "model_id": request.model_id,
            },
        )

        # Validate and prepare source image
        try:
            init_image = base64_to_pil_image(request.init_image)
            init_image = prepare_img2img_image(
                init_image, target_width=request.width, target_height=request.height
            )
        except Exception as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid source image: {str(e)}"
            )

        # Get model manager and ensure model is loaded
        model_manager = get_model_manager()
        if not model_manager.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Model manager not initialized. Try minimal mode for API testing.",
            )

        # Auto-select or switch model if needed
        target_model = request.model_id or model_manager.auto_select_model(
            request.prompt
        )
        if model_manager.current_model != target_model:
            req_logger.info(
                f"Switching model: {model_manager.current_model} → {target_model}"
            )
            switch_success = await model_manager.switch_model(target_model)
            if not switch_success:
                raise HTTPException(
                    status_code=500, detail=f"Failed to switch to model: {target_model}"
                )

        # Handle ControlNet if specified
        controlnet_result = None
        if request.controlnet:
            controlnet_manager = get_controlnet_manager()

            # Prepare control image
            control_image = base64_to_pil_image(request.controlnet.image)

            # Ensure ControlNet pipeline is ready
            if not await controlnet_manager.create_pipeline(
                model_manager.current_model_path, request.controlnet.type  # type: ignore
            ):
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to setup ControlNet: {request.controlnet.type}",
                )

            # Generate with ControlNet
            controlnet_result = await controlnet_manager.generate_with_controlnet(
                prompt=request.prompt,
                control_image=control_image,
                controlnet_type=request.controlnet.type,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                controlnet_strength=request.controlnet.strength,
                seed=request.seed,
            )

            generated_images = controlnet_result["images"]

        else:
            # Standard img2img generation
            generation_params = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "image": init_image,
                "strength": request.strength,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "width": request.width or init_image.width,
                "height": request.height or init_image.height,
                "seed": request.seed,
            }

            # Generate using model manager's img2img capability
            generation_result = await model_manager.generate_img2img(
                **generation_params
            )
            generated_images = generation_result["images"]

        # Process results
        processing_time = time.time() - start_time
        task_id = f"img2img_{int(time.time() * 1000)}"

        # Save images and prepare response
        image_urls = []
        for i, image in enumerate(generated_images):
            image_path = await save_generation_output(
                image, task_id=f"{task_id}_{i}", subfolder="img2img"
            )
            image_urls.append(str(image_path))

        # Prepare metadata
        metadata = {
            "task_id": task_id,
            "request_params": request.model_dump(),
            "generation_params": (
                generation_params
                if not request.controlnet
                else controlnet_result.get("generation_params", {})  # type: ignore
            ),
            "model_used": model_manager.current_model,
            "processing_time": processing_time,
            "image_count": len(generated_images),
            "controlnet_info": (
                {
                    "type": request.controlnet.type,
                    "strength": request.controlnet.strength,
                }
                if request.controlnet
                else None
            ),
            "vram_usage": model_manager.get_vram_usage(),
            "timestamp": time.time(),
        }

        # Save metadata asynchronously
        background_tasks.add_task(
            save_generation_metadata, metadata, task_id, "img2img"
        )

        gen_logger.info(
            "✅ Img2img generation completed",
            extra={
                "task_id": task_id,
                "processing_time": processing_time,
                "image_count": len(generated_images),
                "vram_used": metadata["vram_usage"],
            },
        )

        return {
            "success": True,
            "message": "img2img generation completed successfully",
            "data": {
                "task_id": task_id,
                "images": image_urls,
                "metadata": metadata,
                "processing_time": {
                    "total": processing_time,
                    "generation": processing_time - 1,  # Approximate
                    "preprocessing": 1,
                },
                "model_info": {
                    "model_used": model_manager.current_model,
                    "vram_usage": metadata["vram_usage"],
                },
            },
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        req_logger.error(
            f"❌ Img2img generation failed: {str(e)}",
            extra={
                "error_type": type(e).__name__,
                "processing_time": time.time() - start_time,
            },
        )
        raise HTTPException(
            status_code=500, detail=f"img2img generation failed: {str(e)}"
        )


@router.post("/inpaint", response_model=GenerationResponse)
async def generate_inpaint(
    request: InpaintRequest, background_tasks: BackgroundTasks, http_request: Request
) -> Dict[str, Any]:
    """
    Inpaint masked areas of an image based on text prompt

    Supports:
    - Custom mask blur and fill methods
    - Multiple inpainting strategies
    - Full integration with model management
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    req_logger = get_request_logger(request_id)
    gen_logger = get_generation_logger(request_id, "inpaint")

    start_time = time.time()

    try:
        req_logger.info(
            "🎭 Starting inpaint generation",
            extra={
                "prompt_length": len(request.prompt),
                "strength": request.strength,
                "mask_blur": request.mask_blur,
                "fill_method": request.inpainting_fill,
            },
        )

        # Validate and prepare images
        try:
            init_image = base64_to_pil_image(request.init_image)
            mask_image = base64_to_pil_image(request.mask_image)

            # Prepare inpaint inputs
            init_image, mask_image = prepare_inpaint_mask(
                init_image,
                mask_image,
                blur_radius=request.mask_blur,
                target_width=request.width,
                target_height=request.height,
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid images: {str(e)}")

        # Get model manager
        model_manager = get_model_manager()
        if not model_manager.is_initialized:
            raise HTTPException(status_code=503, detail="Model manager not initialized")

        # Auto-select model for inpainting
        target_model = request.model_id or model_manager.auto_select_model(
            request.prompt, task_type="inpaint"
        )

        # Switch model if needed
        if model_manager.current_model != target_model:
            req_logger.info(f"Switching to inpaint model: {target_model}")
            switch_success = await model_manager.switch_model(target_model)
            if not switch_success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to switch to inpaint model: {target_model}",
                )

        # Prepare generation parameters
        generation_params = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "image": init_image,
            "mask_image": mask_image,
            "strength": request.strength,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "width": request.width or init_image.width,
            "height": request.height or init_image.height,
            "seed": request.seed,
        }

        # Generate inpainted image
        generation_result = await model_manager.generate_inpaint(**generation_params)
        generated_images = generation_result["images"]

        # Process results
        processing_time = time.time() - start_time
        task_id = f"inpaint_{int(time.time() * 1000)}"

        # Save images
        image_urls = []
        for i, image in enumerate(generated_images):
            image_path = await save_generation_output(
                image, task_id=f"{task_id}_{i}", subfolder="inpaint"
            )
            image_urls.append(str(image_path))

        # Prepare metadata
        metadata = {
            "task_id": task_id,
            "request_params": request.model_dump(),
            "generation_params": generation_params,
            "model_used": model_manager.current_model,
            "processing_time": processing_time,
            "image_count": len(generated_images),
            "inpaint_info": {
                "mask_blur": request.mask_blur,
                "fill_method": request.inpainting_fill,
                "strength": request.strength,
            },
            "vram_usage": model_manager.get_vram_usage(),
            "timestamp": time.time(),
        }

        # Save metadata asynchronously
        background_tasks.add_task(
            save_generation_metadata, metadata, task_id, "inpaint"
        )

        gen_logger.info(
            "✅ Inpaint generation completed",
            extra={
                "task_id": task_id,
                "processing_time": processing_time,
                "image_count": len(generated_images),
            },
        )

        return {
            "success": True,
            "message": "Inpaint generation completed successfully",
            "data": {
                "task_id": task_id,
                "images": image_urls,
                "metadata": metadata,
                "processing_time": {
                    "total": processing_time,
                    "generation": processing_time - 1.5,
                    "preprocessing": 1.5,
                },
            },
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        req_logger.error(
            f"❌ Inpaint generation failed: {str(e)}",
            extra={
                "error_type": type(e).__name__,
                "processing_time": time.time() - start_time,
            },
        )
        raise HTTPException(
            status_code=500, detail=f"Inpaint generation failed: {str(e)}"
        )


@router.get("/status")
async def get_img2img_status() -> Dict[str, Any]:
    """Get current img2img service status"""
    model_manager = get_model_manager()
    controlnet_manager = get_controlnet_manager()

    return {
        "success": True,
        "data": {
            "img2img_available": model_manager.is_initialized,
            "current_model": model_manager.current_model,
            "supported_models": list(model_manager.available_models.keys()),
            "controlnet_status": controlnet_manager.get_status(),
            "vram_usage": (
                model_manager.get_vram_usage()
                if model_manager.is_initialized
                else "N/A"
            ),
            "capabilities": {
                "img2img": True,
                "inpaint": True,
                "controlnet": len(controlnet_manager.processors) > 0,
                "supported_controlnet_types": controlnet_manager.supported_types,
            },
        },
        "timestamp": time.time(),
    }


@router.get("/models")
async def list_img2img_models() -> Dict[str, Any]:
    """List available models for img2img generation"""
    model_manager = get_model_manager()

    return {
        "success": True,
        "data": {
            "available_models": [
                {
                    "model_id": model_id,
                    "info": info,
                    "capabilities": {
                        "img2img": True,
                        "inpaint": info.get("supports_inpaint", True),
                        "recommended_for": info.get("strengths", []),
                    },
                }
                for model_id, info in model_manager.available_models.items()
            ],
            "current_model": model_manager.current_model,
            "controlnet_models": list(get_controlnet_manager().processors.keys()),
        },
        "timestamp": time.time(),
    }
