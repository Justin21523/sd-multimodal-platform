# app/api/v1/txt2img.py
"""
SD Multi-Modal Platform - txt2img API Router
This module provides the API endpoints for text-to-image generation.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import Settings, settings, get_settings
from app.schemas.requests import Txt2ImgRequest, PromptTemplate, BUILTIN_TEMPLATES
from app.schemas.responses import (
    Txt2ImgResponse,
    GeneratedImage,
    ImageMetadata,
    ErrorResponse,
)
from services.generation.txt2img_service import Txt2ImgService
from utils.logging_utils import get_request_logger


# Create API router
router = APIRouter()
logger = logging.getLogger(__name__)

# Global txt2img service instance
_txt2img_service = None


async def get_txt2img_service() -> Txt2ImgService:
    """Get the global txt2img service instance, initializing if necessary"""
    global _txt2img_service

    if _txt2img_service is None:
        _txt2img_service = Txt2ImgService()
        await _txt2img_service.initialize()

    return _txt2img_service


@router.post("/txt2img", response_model=Txt2ImgResponse)
async def generate_image(
    request: Txt2ImgRequest,
    background_tasks: BackgroundTasks,
    service: Txt2ImgService = Depends(get_txt2img_service),
    config: Settings = Depends(get_settings),
) -> Union[Txt2ImgResponse, JSONResponse]:
    """
    Handle text-to-image generation requests.

    Feattures:
    - Supports custom prompts and negative prompts
    - Allows for custom generation parameters
    - Returns generated images with metadata
    - Handles errors gracefully with detailed logging
    - Supports background tasks for cleanup
    - Provides prompt templates for easier prompt management
    """
    start_time = time.time()

    # Generate a unique task ID based on the current time and prompt hash
    task_id = f"txt2img_{int(time.time() * 1000)}_{request.get_prompt_hash()}"

    # Initialize request logger
    request_logger = get_request_logger(task_id)
    request_logger.info(
        f"🎨 Starting txt2img generation",
        extra={
            "task_id": task_id,
            "prompt_length": len(request.prompt),
            "model_id": request.model_id,
            "dimensions": (
                f"{request.generation_params.width}x{request.generation_params.height}"
                if request.generation_params
                else "default"
            ),
            "steps": (
                request.generation_params.num_inference_steps
                if request.generation_params
                else config.DEFAULT_STEPS
            ),
            "seed": (
                request.generation_params.seed
                if request.generation_params
                else "random"
            ),
        },
    )

    try:
        # === Parameter Validation ===
        effective_params = request.get_effective_params()

        # Record the effective parameters for logging
        request_logger.debug(f"Generation parameters: {effective_params}")

        # Check if the model ID is valid
        model_load_start = time.time()

        # Check if model_id is provided, otherwise use primary model
        current_model = await service.ensure_model_loaded(request.model_id)

        model_load_time = time.time() - model_load_start

        request_logger.info(
            f"✅ Model ready: {current_model}",
            extra={"model": current_model, "load_time": model_load_time},
        )

        # === Image Generation ===1
        generation_start = time.time()

        # Generate the image using the service
        generation_result = await service.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            **effective_params,
        )

        generation_time = time.time() - generation_start

        request_logger.info(
            f"🖼️  Image generation completed",
            extra={
                "generation_time": generation_time,
                "output_path": generation_result.get("image_path"),
                "vram_used": generation_result.get("vram_used", "unknown"),
            },
        )

        # === Metadata Handling ===
        save_start = time.time()

        # Create metadata object
        metadata = ImageMetadata(
            seed=effective_params["seed"],
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            model=current_model,
            model_hash=generation_result.get("model_hash"),
            width=effective_params["width"],
            height=effective_params["height"],
            steps=effective_params["num_inference_steps"],
            cfg_scale=effective_params["guidance_scale"],
            sampler=effective_params.get("sampler", "default"),
            scheduler=effective_params.get("scheduler", config.DEFAULT_SCHEDULER),
            generation_time=generation_time,
            device=config.DEVICE,
            vram_used=generation_result.get("vram_used"),
            peak_memory=generation_result.get("peak_memory"),
            filename=Path(generation_result["image_path"]).name,
            file_size=generation_result.get("file_size"),
        )

        # Save metadata if requested
        if request.save_metadata:
            metadata_path = await service.save_metadata(metadata, task_id)
            request_logger.debug(f"Metadata saved: {metadata_path}")

        # Record the time taken to save metadata

        # Generate the image URL
        image_url = f"/outputs/txt2img/{Path(generation_result['image_path']).name}"

        # Create the GeneratedImage response object
        generated_image = GeneratedImage(
            url=image_url,
            filename=Path(generation_result["image_path"]).name,
            local_path=generation_result["image_path"],
            metadata=metadata,
            width=effective_params["width"],
            height=effective_params["height"],
            seed=effective_params["seed"],
        )

        # If base64 encoding is requested, convert the image to base64
        if request.return_base64:
            generated_image.base64_data = await service.get_image_base64(
                generation_result["image_path"]
            )

        # Calculate total time taken for the request
        total_time = time.time() - start_time

        # Create the success response
        response = Txt2ImgResponse.success_response(
            images=[generated_image],
            task_id=task_id,
            total_time=total_time,
            generation_time=generation_time,
            model_used=current_model,
            device_used=config.DEVICE,
            model_load_time=model_load_time if model_load_time > 0.1 else None,
        )

        # If cleanup is needed, add it as a background task
        if config.KEEP_GENERATIONS_DAYS > 0:
            background_tasks.add_task(
                service.cleanup_old_files, days=config.KEEP_GENERATIONS_DAYS
            )

        request_logger.info(
            f"✅ txt2img completed successfully",
            extra={
                "task_id": task_id,
                "total_time": total_time,
                "output_file": generated_image.filename,
            },
        )

        return response

    except Exception as exc:
        logger.error(f"Failed to get service status: {exc}")

        # Log the error with detailed information
        total_time = time.time() - start_time

        # Log the error with detailed information
        request_logger.error(
            f"❌ txt2img generation failed: {str(exc)}",
            extra={
                "task_id": task_id,
                "error_type": type(exc).__name__,
                "total_time": total_time,
            },
            exc_info=True,
        )

        # Determine the error code and status based on the exception type
        msg = str(exc)
        if isinstance(exc, ValueError):
            # Validation error (e.g., invalid parameters)
            error_code = "VALIDATION_ERROR"
            status_code = 422
        elif "CUDA out of memory" in msg or "OutOfMemoryError" in msg:
            # CUDA out of memory error
            error_code = "OUT_OF_MEMORY"
            status_code = 507  # Insufficient Storage
        elif "model" in msg.lower() and "not found" in msg.lower():
            # Model not found error
            error_code = "MODEL_NOT_FOUND"
            status_code = 404
        else:
            # General generation error
            error_code = "GENERATION_ERROR"
            status_code = 500

        # Create the error response
        error_response = Txt2ImgResponse.error_response(
            error_message=msg,
            task_id=task_id,
            error_code=error_code,
            model_used=getattr(service, "current_model", "unknown"),
            device_used=config.DEVICE,
        )

        # If cleanup is needed, add it as a background task
        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(),  # pydantic v2 -> model_dump()
        )


@router.get("/txt2img/templates")
async def get_prompt_templates() -> Dict[str, Any]:
    """Retrieve available prompt templates for text-to-image generation"""

    return {
        "success": True,
        "templates": [template.model_dump() for template in BUILTIN_TEMPLATES],
        "total_count": len(BUILTIN_TEMPLATES),
        "message": "Available prompt templates retrieved successfully",
    }


@router.post("/txt2img/preview")
async def preview_prompt_with_template(
    prompt: str, template_name: str = "photorealistic"
) -> Dict[str, Any]:
    """Use a prompt template to preview the processed prompt"""

    # Validate the template name
    template = None
    for t in BUILTIN_TEMPLATES:
        if t.name == template_name:
            template = t
            break

    if not template:
        raise HTTPException(
            status_code=404, detail=f"Template '{template_name}' not found"
        )

    # Apply the template to the prompt
    processed_positive, processed_negative = template.apply_to_prompt(prompt)

    return {
        "success": True,
        "data": {
            "original_prompt": prompt,
            "template_used": template_name,
            "processed_positive": processed_positive,
            "processed_negative": processed_negative,
            "template_info": template.dict(),
        },
        "message": "Prompt preview generated successfully",
    }


@router.get("/txt2img/status")
async def get_generation_status(
    service: Txt2ImgService = Depends(get_txt2img_service),
    config: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Get the current status of the txt2img service"""

    try:
        status_info = await service.get_status()

        return {
            "success": True,
            "status": "ready",
            "data": {
                "current_model": status_info.get("current_model"),
                "model_loaded": status_info.get("model_loaded", False),
                "device": config.DEVICE,
                "memory_info": status_info.get("memory_info"),
                "recent_generations": status_info.get("recent_count", 0),
                "average_generation_time": status_info.get("avg_time"),
                "service_uptime": status_info.get("uptime"),
            },
            "message": "Service status retrieved successfully",
        }

    except Exception as exc:
        logger.error(f"Failed to get service status: {exc}")

        return {
            "success": False,
            "status": "error",
            "error": str(exc),
            "message": "Failed to retrieve service status",
        }


@router.get("/txt2img/status")
async def get_service_status() -> Dict[str, Any]:
    """取得服務狀態"""

    return {
        "success": True,
        "status": "ready",
        "data": {
            "service": "txt2img",
            "version": "1.0.0-phase1",
            "device": settings.DEVICE,
            "primary_model": settings.PRIMARY_MODEL,
            "torch_dtype": settings.TORCH_DTYPE,
            "service_features": [
                "Type-safe pipeline result handling",
                "Multiple image format support",
                "Unified PIL.Image extraction",
                "Memory optimization ready",
            ],
            "supported_result_types": [
                "StableDiffusionXLPipelineOutput",
                "StableDiffusionPipelineOutput",
                "List[PIL.Image]",
                "List[np.ndarray]",
                "np.ndarray (batch)",
                "PyTorch tensors",
            ],
        },
        "timestamp": time.time(),
    }


@router.get("/txt2img/test-types")
async def test_type_handling() -> Dict[str, Any]:
    """測試型別處理邏輯 (開發用)"""

    return {
        "success": True,
        "message": "Type handling test endpoint",
        "data": {
            "type_safety_features": {
                "pipeline_result_extraction": "✅ Unified _extract_images_from_result()",
                "pil_image_validation": "✅ All images converted to PIL.Image",
                "numpy_array_support": "✅ Handles both batch and single arrays",
                "tensor_conversion": "✅ PyTorch tensor to PIL.Image conversion",
                "error_recovery": "✅ Individual image processing with fallback",
                "memory_cleanup": "✅ GPU memory management on errors",
            },
            "supported_formats": {
                "input_types": [
                    "PipelineOutput.images",
                    "List[PIL.Image]",
                    "List[np.ndarray]",
                    "np.ndarray (H,W,C) or (B,H,W,C)",
                    "torch.Tensor",
                ],
                "output_guarantee": "Always List[PIL.Image]",
                "fallback_strategy": "Skip invalid images, log warnings",
            },
        },
        "timestamp": time.time(),
    }
