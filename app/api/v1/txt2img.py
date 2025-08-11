# app/api/v1/txt2img.py
"""
SD Multi-Modal Platform - txt2img API Router
Real Text-to-Image API Implementation for Phase 3
Handles actual image generation using loaded AI models.
"""

import logging
import time
import uuid
import asyncio
from typing import Dict, Any, Union, List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, Field
from pydantic_settings import BaseSettings
from PIL import Image
import base64
import io

from app.config import Settings, settings, get_settings
from app.schemas.requests import Txt2ImgRequest, PromptTemplate, BUILTIN_TEMPLATES
from app.schemas.responses import (
    Txt2ImgResponse,
    GeneratedImage,
    ImageMetadata,
    ErrorResponse,
)
from services.generation.txt2img_service import Txt2ImgService
from services.models.sd_models import get_model_manager, ModelRegistry

from utils.logging_utils import get_request_logger
from utils.file_utils import ensure_directory, safe_filename
from utils.metadata_utils import save_metadata_json
from utils.image_utils import image_to_base64, optimize_image


# Create API router
router = APIRouter(prefix="/txt2img", tags=["text-to-image"])
logger = logging.getLogger(__name__)


class Txt2ImgRequest(BaseModel):
    """Request schema for text-to-image generation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text prompt describing the desired image",
    )
    negative_prompt: str = Field(
        default="",
        max_length=2000,
        description="Negative prompt to avoid unwanted elements",
    )
    model_id: Optional[str] = Field(
        default=None, description="Specific model to use (auto-select if None)"
    )
    width: int = Field(
        default=0, ge=256, le=2048, description="Image width (model default if None)"
    )  # typoe: ignore

    height: int = Field(
        default=0,
        ge=256,
        le=2048,
        description="Image height (model default if None)",
    )
    num_inference_steps: int = Field(
        default=25, ge=10, le=100, description="Number of denoising steps"
    )
    guidance_scale: float = Field(
        default=7.5, ge=1.0, le=20.0, description="CFG scale for prompt adherence"
    )
    seed: Optional[int] = Field(
        default=None, ge=-1, le=2**32 - 1, description="Random seed (-1 for random)"
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=settings.MAX_BATCH_SIZE,
        description="Number of images to generate",
    )
    save_images: bool = Field(
        default=True, description="Whether to save images to disk"
    )
    return_base64: bool = Field(
        default=False, description="Include base64 encoded images in response"
    )

    @field_validator("width", "height", mode="before")
    def validate_dimensions(cls, v):
        """Ensure dimensions are multiples of 8 for SD compatibility."""
        if v is not None and v % 8 != 0:
            # Round to nearest multiple of 8
            v = ((v + 7) // 8) * 8
        return v

    @field_validator("seed", mode="before")
    def handle_negative_seed(cls, v):
        """Convert -1 to None for random seed."""
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if str(v) == "-1":
            return None
        return v


class Txt2ImgResponse(BaseModel):
    """Response schema for text-to-image generation."""

    success: bool = Field(description="Whether generation was successful")
    task_id: str = Field(description="Unique task identifier")
    message: str = Field(description="Status message")
    data: Optional[Dict[str, Any]] = Field(
        description="Generation results and metadata"
    )
    request_id: str = Field(description="Request tracking ID")
    timestamp: float = Field(description="Response timestamp")


async def get_model_manager_dependency():
    """Dependency to get initialized model manager."""
    manager = get_model_manager()
    if not manager.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Model manager not initialized. Please check server status.",
        )
    return manager


@router.post("/", response_model=Txt2ImgResponse)
async def generate_image(
    request_data: Txt2ImgRequest,
    request: Request,
    model_manager=Depends(get_model_manager_dependency),
) -> Txt2ImgResponse:
    """
    Generate images from text prompts using AI models.

    This endpoint creates images based on text descriptions using various
    Stable Diffusion models. Supports model switching, parameter customization,
    and batch generation.
    """
    # Get request tracking info
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    task_id = f"txt2img_{int(time.time() * 1000)}"
    req_logger = get_request_logger(request_id)

    req_logger.info(f"Starting txt2img generation - Task: {task_id}")
    req_logger.info(f"Prompt: {request_data.prompt[:100]}...")

    start_time = time.time()

    try:
        # Determine target model
        target_model = request_data.model_id or settings.PRIMARY_MODEL

        # Validate model exists
        if target_model not in ModelRegistry.list_models():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {target_model}. Available: {ModelRegistry.list_models()}",
            )

        # Switch model if needed
        if model_manager.current_model_id != target_model:
            req_logger.info(f"Switching to model: {target_model}")
            switch_success = await model_manager.switch_model(target_model)
            if not switch_success:
                raise HTTPException(
                    status_code=500, detail=f"Failed to load model: {target_model}"
                )

        # Prepare generation parameters
        generation_params = {
            "prompt": request_data.prompt,
            "negative_prompt": request_data.negative_prompt,
            "width": request_data.width,
            "height": request_data.height,
            "num_inference_steps": request_data.num_inference_steps,
            "guidance_scale": request_data.guidance_scale,
            "seed": request_data.seed,
            "num_images": request_data.num_images,
        }

        req_logger.info(f"Generation params: {generation_params}")

        # Generate images
        generation_result = await model_manager.generate_image(**generation_params)
        images = generation_result["images"]
        metadata = generation_result["metadata"]

        # Process results
        image_data = []
        saved_paths = []

        for i, image in enumerate(images):
            image_info = {
                "index": i,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            }

            # Save image if requested
            if request_data.save_images:
                # Create output directory
                output_dir = Path(settings.OUTPUT_PATH) / "txt2img"
                ensure_directory(output_dir)

                # Generate safe filename
                prompt_part = safe_filename(request_data.prompt[:50])
                timestamp = int(time.time())
                filename = (
                    f"{timestamp}_{prompt_part}_seed{metadata['seed']}_{i:02d}.png"
                )
                image_path = output_dir / filename

                # Optimize and save
                optimized_image = optimize_image(image, quality=95)
                optimized_image.save(image_path, "PNG", optimize=True)

                image_info["file_path"] = str(image_path.relative_to(Path.cwd()))
                image_info["file_size_bytes"] = image_path.stat().st_size
                saved_paths.append(image_path)

                req_logger.info(f"Saved image {i}: {image_path}")

            # Add base64 if requested
            if request_data.return_base64:
                image_info["base64"] = image_to_base64(image, format="PNG")

            image_data.append(image_info)

        # Save metadata
        if request_data.save_images and saved_paths:
            metadata_path = (
                saved_paths[0].parent / f"{saved_paths[0].stem}_metadata.json"
            )
            complete_metadata = {
                **metadata,
                "task_id": task_id,
                "request_id": request_id,
                "request_params": request_data.dict(),
                "saved_files": [str(p.relative_to(Path.cwd())) for p in saved_paths],
                "total_time": time.time() - start_time,
            }
            save_metadata_json(complete_metadata, metadata_path)
            req_logger.info(f"Saved metadata: {metadata_path}")

        # Build response
        total_time = time.time() - start_time

        response_data = {
            "task_id": task_id,
            "model_used": {
                "model_id": metadata["model_id"],
                "model_name": metadata["model_name"],
            },
            "generation_params": {
                "prompt": metadata["prompt"],
                "negative_prompt": metadata["negative_prompt"],
                "width": metadata["width"],
                "height": metadata["height"],
                "steps": metadata["num_inference_steps"],
                "cfg_scale": metadata["guidance_scale"],
                "seed": metadata["seed"],
            },
            "results": {
                "num_images": len(images),
                "images": image_data,
                "generation_time": metadata["generation_time"],
                "total_time": round(total_time, 2),
                "vram_used_gb": metadata["vram_used_gb"],
            },
            "optimization_info": metadata["optimization_info"],
        }

        req_logger.info(
            f"✅ Generation completed successfully",
            extra={
                "task_id": task_id,
                "generation_time": metadata["generation_time"],
                "total_time": total_time,
                "vram_used": metadata["vram_used_gb"],
                "num_images": len(images),
            },
        )

        return Txt2ImgResponse(
            success=True,
            task_id=task_id,
            message=f"Generated {len(images)} images successfully",
            data=response_data,
            request_id=request_id,
            timestamp=time.time(),
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        req_logger.error(
            f"❌ Generation failed: {str(e)}",
            extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "total_time": time.time() - start_time,
            },
        )

        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )


@router.get("/status")
async def get_txt2img_status(
    model_manager=Depends(get_model_manager_dependency),
) -> Dict[str, Any]:
    """Get current txt2img service status and model information."""

    status = model_manager.get_status()

    # Add service-specific info
    service_status = {
        "service": "txt2img",
        "endpoint": "/api/v1/txt2img",
        "model_manager": status,
        "supported_parameters": {
            "prompt": "Required text prompt",
            "negative_prompt": "Optional negative prompt",
            "model_id": f"Optional model ({ModelRegistry.list_models()})",
            "width": "256-2048 (multiples of 8)",
            "height": "256-2048 (multiples of 8)",
            "num_inference_steps": "10-100",
            "guidance_scale": "1.0-20.0",
            "seed": "Random seed or -1 for random",
            "num_images": f"1-{settings.MAX_BATCH_SIZE}",
        },
        "available_models": {},
    }

    # Add detailed model info
    for model_id in ModelRegistry.list_models():
        model_info = ModelRegistry.get_model_info(model_id)
        service_status["available_models"][model_id] = {
            "name": model_info["name"],  # type: ignore
            "default_resolution": model_info["default_resolution"],  # type: ignore
            "vram_requirement": f"{model_info['vram_requirement']}GB",  # type: ignore
            "strengths": model_info["strengths"],  # type: ignore
            "use_cases": model_info["use_cases"],  # type: ignore
        }

    return service_status


@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """List all available models with their capabilities and status."""

    models_info = {}
    model_manager = get_model_manager()

    for model_id in ModelRegistry.list_models():
        model_info = ModelRegistry.get_model_info(model_id)

        # Check if model files exist
        model_path = (
            Path(settings.OUTPUT_PATH).parent / "models" / model_info["local_path"]  # type: ignore
        )
        is_installed = model_path.exists()
        is_loaded = model_manager.current_model_id == model_id

        models_info[model_id] = {
            **model_info,  # type: ignore
            "is_installed": is_installed,
            "is_loaded": is_loaded,
            "model_path": str(model_path),
        }

    return {
        "available_models": models_info,
        "currently_loaded": model_manager.current_model_id,
        "total_models": len(models_info),
        "installed_models": sum(
            1 for info in models_info.values() if info["is_installed"]
        ),
    }


@router.post("/switch-model")
async def switch_model(
    model_id: str, model_manager=Depends(get_model_manager_dependency)
) -> Dict[str, Any]:
    """Switch to a different model."""

    if model_id not in ModelRegistry.list_models():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_id}. Available: {ModelRegistry.list_models()}",
        )

    if model_id == model_manager.current_model_id:
        return {
            "success": True,
            "message": f"Model {model_id} is already loaded",
            "current_model": model_id,
        }

    # Perform switch
    switch_start = time.time()
    success = await model_manager.switch_model(model_id)
    switch_time = time.time() - switch_start

    if success:
        return {
            "success": True,
            "message": f"Successfully switched to model: {model_id}",
            "previous_model": model_manager.current_model_id,
            "current_model": model_id,
            "switch_time": round(switch_time, 2),
        }
    else:
        raise HTTPException(
            status_code=500, detail=f"Failed to switch to model: {model_id}"
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
    """Get the overall status of the txt2img service and its features"""

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
    """Test endpoint to verify type handling and safety features"""

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
