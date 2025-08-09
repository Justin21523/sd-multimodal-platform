# backend/api/v1/endpoints/models.py
"""
Model Management Endpoints

Handles model switching, information, and status operations.
"""

from fastapi import APIRouter, HTTPException
import torch
import psutil
import logging

from backend.schemas.requests import ModelSwitchRequest, SchedulerChangeRequest
from backend.schemas.responses import ModelInfoResponse, StatusResponse
from backend.core.sd_pipeline import sd_manager
from backend.core.model_loader import model_loader
from backend.config.model_config import ModelRegistry


router = APIRouter(prefix="/models", tags=["Model Management"])
logger = logging.getLogger(__name__)


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get current model information and available models

    Returns details about the currently loaded model and
    lists all available models for switching.
    """
    try:
        pipeline_info = sd_manager.get_pipeline_info()
        available_models = ModelRegistry.list_models()

        return ModelInfoResponse(
            success=True, model_info=pipeline_info, available_models=available_models
        )

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/switch", response_model=StatusResponse)
async def switch_model(request: ModelSwitchRequest):
    """
    Switch to a different Stable Diffusion model

    Args:
        request: Model switch parameters

    Returns:
        Status of the model switch operation
    """
    try:
        logger.info(f"Switching to model: {request.model_id}")

        # Validate model exists
        try:
            ModelRegistry.get_model_config(request.model_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Check if model is available locally
        availability = model_loader.check_model_availability(request.model_id)

        if not availability.get("available_locally", False):
            # Attempt to download model
            logger.info(f"Model not found locally, downloading: {request.model_id}")

            if not model_loader.download_model_if_needed(request.model_id):
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download model: {request.model_id}",
                )

        # Unload current pipeline if force reload
        if request.force_reload and sd_manager.pipeline is not None:
            sd_manager.unload_pipeline()

        # Update manager model ID
        sd_manager.model_id = request.model_id
        sd_manager.model_config = ModelRegistry.get_model_config(request.model_id)

        # Load new pipeline
        if not sd_manager.load_pipeline(force_reload=True):
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {request.model_id}"
            )

        logger.info(f"✅ Successfully switched to model: {request.model_id}")

        return StatusResponse(
            success=True,
            message=f"Successfully switched to model: {request.model_id}",
            data={
                "model_id": request.model_id,
                "model_name": sd_manager.model_config.name,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler", response_model=StatusResponse)
async def change_scheduler(request: SchedulerChangeRequest):
    """
    Change the diffusion scheduler algorithm

    Args:
        request: Scheduler change parameters

    Returns:
        Status of the scheduler change operation
    """
    try:
        if not sd_manager.set_scheduler(request.scheduler):
            raise HTTPException(
                status_code=500, detail=f"Failed to set scheduler: {request.scheduler}"
            )

        return StatusResponse(
            success=True,
            message=f"Scheduler changed to: {request.scheduler}",
            data={"scheduler": request.scheduler},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scheduler change failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_model_status():
    """
    Get detailed model and system status

    Returns comprehensive information about model loading status,
    memory usage, and system capabilities.
    """
    try:
        pipeline_info = sd_manager.get_pipeline_info()

        # Memory information
        memory_info = {
            "system_ram": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "used_percent": psutil.virtual_memory().percent,
            }
        }

        if torch.cuda.is_available():
            memory_info["gpu"] = {
                "device_name": torch.cuda.get_device_name(),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory
                / (1024**3),
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "cached_gb": torch.cuda.memory_reserved() / (1024**3),
            }

        return {
            "success": True,
            "model_status": pipeline_info,
            "memory_info": memory_info,
            "available_models": ModelRegistry.list_models(),
            "system_info": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else None
                ),
                "pytorch_version": torch.__version__,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
