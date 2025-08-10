# backend/api/v1/endpoints/txt2img.py
"""
Text-to-Image Generation Endpoints

Provides REST API endpoints for text-to-image generation with:
- Input validation and sanitization
- Error handling and recovery
- Response formatting and metadata
"""

import time
import asyncio
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import logging

from backend.schemas.requests import Text2ImageRequest
from backend.schemas.responses import Text2ImageResponse, ImageData, ErrorResponse
from backend.core.sd_pipeline import sd_manager
from backend.core.image_utils import ImageProcessor
from backend.config.settings import Settings

router = APIRouter(prefix="/txt2img", tags=["Text-to-Image"])
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=Text2ImageResponse)
async def generate_text_to_image(
    request: Text2ImageRequest, background_tasks: BackgroundTasks
) -> Text2ImageResponse:
    """
    Generate images from text prompts

    This endpoint accepts a text description and generates corresponding images
    using Stable Diffusion models. Supports various parameters for controlling
    the generation process.

    Args:
        request: Text2Image generation parameters
        background_tasks: For cleanup operations

    Returns:
        Response containing generated images and metadata

    Raises:
        HTTPException: For validation errors or generation failures
    """
    start_time = time.time()

    try:
        logger.info(f"Generating image for prompt: '{request.prompt[:50]}...'")

        # Validate and prepare parameters
        generation_params = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width or Settings.default_width,
            "height": request.height or Settings.default_height,
            "num_inference_steps": request.num_inference_steps
            or Settings.default_steps,
            "guidance_scale": request.guidance_scale or Settings.default_guidance_scale,
            "seed": request.seed,
            "batch_size": request.batch_size or 1,
        }

        # Change scheduler if requested
        if request.scheduler and request.scheduler != "dpm":
            if not sd_manager.set_scheduler(request.scheduler):
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to set scheduler: {request.scheduler}",
                )

        # Generate images
        images = await asyncio.get_event_loop().run_in_executor(
            None, sd_manager.generate_image, **generation_params
        )
        # Image Procerssor instance
        image_processor = ImageProcessor()
        # Check if images were generated
        if not images:
            raise HTTPException(status_code=500, detail="Image generation failed")

        # Prepare metadata for saving
        metadata = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": generation_params["width"],
            "height": generation_params["height"],
            "steps": generation_params["num_inference_steps"],
            "guidance_scale": generation_params["guidance_scale"],
            "seed": generation_params["seed"],
            "scheduler": request.scheduler or "dpm",
            "model": sd_manager.model_id,
            "timestamp": time.time(),
        }

        # Save images and prepare response
        if len(images) == 1:
            filename, full_path = image_processor.save_image_with_metadata(
                image=images[0], metadata=metadata
            )
            saved_results = [(filename, full_path)]
        else:
            saved_results = image_processor.save_batch_images(
                images=images, metadata=metadata
            )

        # Create response data
        image_data_list = []
        for i, (filename, full_path) in enumerate(saved_results):
            image_data = ImageData(
                filename=filename,
                url=f"/api/v1/images/{filename}",
                width=generation_params["width"],
                height=generation_params["height"],
                seed=generation_params["seed"],
            )
            image_data_list.append(image_data)

        generation_time = time.time() - start_time

        # Schedule cleanup if needed
        background_tasks.add_task(image_processor.cleanup_temp_files)

        logger.info(f"✅ Generated {len(images)} image(s) in {generation_time:.2f}s")

        return Text2ImageResponse(
            success=True,
            message=f"Successfully generated {len(images)} image(s)",
            images=image_data_list,
            generation_time=generation_time,
            parameters=generation_params,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )


@router.get("/parameters")
async def get_generation_parameters():
    """
    Get available generation parameters and their valid ranges

    Returns information about supported parameters, schedulers,
    and current model capabilities.
    """
    try:
        pipeline_info = sd_manager.get_pipeline_info()

        return {
            "success": True,
            "parameters": {
                "width": {
                    "min": 64,
                    "max": 2048,
                    "default": Settings.default_width,
                    "step": 8,
                },
                "height": {
                    "min": 64,
                    "max": 2048,
                    "default": Settings.default_height,
                    "step": 8,
                },
                "num_inference_steps": {
                    "min": 1,
                    "max": 100,
                    "default": Settings.default_steps,
                },
                "guidance_scale": {
                    "min": 1.0,
                    "max": 20.0,
                    "default": Settings.default_guidance_scale,
                },
                "batch_size": {"min": 1, "max": Settings.max_batch_size, "default": 1},
            },
            "schedulers": ["dpm", "euler_a", "ddim"],
            "current_model": pipeline_info,
            "formats": ["PNG", "JPEG", "WEBP"],
        }

    except Exception as e:
        logger.error(f"Failed to get parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))
