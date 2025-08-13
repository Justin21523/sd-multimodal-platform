# app/api/v1/queue.py
"""
Queue Management API endpoints
Provides task submission, status tracking, and queue management
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from services.queue.task_manager import get_task_manager, TaskStatus
from app.schemas.requests import Txt2ImgRequest, Img2ImgRequest
from app.schemas.responses import BaseResponse
from utils.logging_utils import get_request_logger

router = APIRouter(prefix="/queue", tags=["Queue Management"])


# Request schemas for queue operations
class PostprocessRequest(BaseModel):
    """Post-processing task request"""

    image_paths: List[str] = Field(..., description="Paths to images for processing")
    pipeline_type: str = Field(
        default="standard", description="Pipeline type: standard/fast/quality"
    )
    steps: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Custom pipeline steps"
    )


class BatchGenerationRequest(BaseModel):
    """Batch generation request"""

    task_type: str = Field(..., description="Type of generation: txt2img/img2img")
    batch_params: List[Dict[str, Any]] = Field(
        ..., description="List of generation parameters"
    )
    postprocess_chain: Optional[List[str]] = Field(
        default=None, description="Post-processing steps"
    )

    class Config:
        schema_extra = {
            "example": {
                "task_type": "txt2img",
                "batch_params": [
                    {"prompt": "cat", "seed": 123},
                    {"prompt": "dog", "seed": 456},
                    {"prompt": "bird", "seed": 789},
                ],
                "postprocess_chain": ["upscale", "face_restore"],
            }
        }


# Response schemas
class TaskSubmissionResponse(BaseResponse):
    """Task submission response"""

    data: Dict[str, str] = Field(..., description="Task submission data")


class TaskStatusResponse(BaseResponse):
    """Task status response"""

    data: Dict[str, Any] = Field(..., description="Task status and progress")


class QueueStatsResponse(BaseResponse):
    """Queue statistics response"""

    data: Dict[str, Any] = Field(..., description="Queue statistics")


@router.post("/submit/txt2img", response_model=TaskSubmissionResponse)
async def submit_txt2img_task(
    request: Txt2ImgRequest,
    postprocess_chain: Optional[List[str]] = Query(default=None),
    req_logger=get_request_logger("queue_submit"),
):
    """Submit text-to-image generation task to queue"""

    try:
        task_manager = get_task_manager()

        # Convert request to generation parameters
        generation_params = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "model_id": request.model_id,
            "width": request.width,
            "height": request.height,
            "steps": request.num_inference_steps,
            "cfg_scale": request.guidance_scale,
            "seed": request.seed,
        }

        # Submit task
        task_id = await task_manager.submit_generation_task(
            task_type="txt2img",
            generation_params=generation_params,
            postprocess_chain=postprocess_chain,
        )

        req_logger.info(f"Submitted txt2img task: {task_id}")

        return TaskSubmissionResponse(
            success=True,
            message="Task submitted to queue",
            data={
                "task_id": task_id,
                "task_type": "txt2img",
                "postprocess_enabled": bool(postprocess_chain),  # type: ignore
            },
        )

    except Exception as e:
        req_logger.error(f"Failed to submit txt2img task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit/img2img", response_model=TaskSubmissionResponse)
async def submit_img2img_task(
    request: Img2ImgRequest,
    postprocess_chain: Optional[List[str]] = Query(default=None),
    req_logger=get_request_logger("queue_submit"),
):
    """Submit image-to-image generation task to queue"""

    try:
        task_manager = get_task_manager()

        # Convert request to generation parameters
        generation_params = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "init_image": request.init_image,
            "model_id": request.model_id,
            "width": request.width,
            "height": request.height,
            "steps": request.num_inference_steps,
            "cfg_scale": request.guidance_scale,
            "strength": request.strength,
            "seed": request.seed,
        }

        # Submit task
        task_id = await task_manager.submit_generation_task(
            task_type="img2img",
            generation_params=generation_params,
            postprocess_chain=postprocess_chain,
        )

        req_logger.info(f"Submitted img2img task: {task_id}")

        return TaskSubmissionResponse(
            success=True,
            message="Task submitted to queue",
            data={
                "task_id": task_id,
                "task_type": "img2img",
                "postprocess_enabled": bool(postprocess_chain),  # type: ignore
            },
        )

    except Exception as e:
        req_logger.error(f"Failed to submit img2img task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit/postprocess", response_model=TaskSubmissionResponse)
async def submit_postprocess_task(
    request: PostprocessRequest, req_logger=get_request_logger("queue_submit")
):
    """Submit post-processing task to queue"""

    try:
        task_manager = get_task_manager()

        # Prepare post-processing parameters
        postprocess_params = {
            "image_paths": request.image_paths,
            "pipeline_type": request.pipeline_type,
            "steps": request.steps,
        }

        # Submit task
        task_id = await task_manager.submit_generation_task(
            task_type="postprocess", generation_params=postprocess_params
        )

        req_logger.info(f"Submitted postprocess task: {task_id}")

        return TaskSubmissionResponse(
            success=True,
            message="Post-processing task submitted to queue",
            data={
                "task_id": task_id,
                "task_type": "postprocess",
                "images_count": len(request.image_paths),  # type: ignore
                "pipeline_type": request.pipeline_type,
            },
        )

    except Exception as e:
        req_logger.error(f"Failed to submit postprocess task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit/batch", response_model=TaskSubmissionResponse)
async def submit_batch_task(
    request: BatchGenerationRequest, req_logger=get_request_logger("queue_submit")
):
    """Submit batch generation task to queue"""

    try:
        task_manager = get_task_manager()

        # Validate batch size
        if len(request.batch_params) > 10:  # Configurable limit
            raise HTTPException(
                status_code=400, detail="Batch size exceeds maximum limit of 10"
            )

        # Submit batch task
        batch_id = await task_manager.submit_batch_task(
            task_type=request.task_type,
            batch_params=request.batch_params,
            postprocess_chain=request.postprocess_chain,
        )

        req_logger.info(f"Submitted batch task: {batch_id}")

        return TaskSubmissionResponse(
            success=True,
            message="Batch task submitted to queue",
            data={
                "batch_id": batch_id,
                "task_type": "batch",
                "batch_size": len(request.batch_params),  # type: ignore
                "postprocess_enabled": bool(request.postprocess_chain),
            },
        )

    except Exception as e:
        req_logger.error(f"Failed to submit batch task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, req_logger=get_request_logger("queue_status")):
    """Get status and progress of a specific task"""

    try:
        task_manager = get_task_manager()
        task_info = await task_manager.get_task_status(task_id)

        if not task_info:
            raise HTTPException(status_code=404, detail="Task not found")

        # Convert TaskInfo to response format
        task_data = {
            "task_id": task_info.task_id,
            "task_type": task_info.task_type,
            "status": task_info.status.value,
            "progress": task_info.progress,
            "created_at": (
                task_info.created_at.isoformat() if task_info.created_at else None
            ),
            "started_at": (
                task_info.started_at.isoformat() if task_info.started_at else None
            ),
            "completed_at": (
                task_info.completed_at.isoformat() if task_info.completed_at else None
            ),
            "error_message": task_info.error_message,
            "result_data": task_info.result_data,
            "meta": task_info.meta,
        }

        # Calculate estimated time remaining
        if task_info.status == TaskStatus.PROCESSING and task_info.started_at:
            elapsed = (task_info.created_at - task_info.started_at).total_seconds()
            if task_info.progress > 0:
                estimated_total = elapsed / task_info.progress
                estimated_remaining = estimated_total - elapsed
                task_data["estimated_remaining_seconds"] = max(0, estimated_remaining)

        return TaskStatusResponse(
            success=True,
            message=f"Task status: {task_info.status.value}",
            data=task_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        req_logger.error(f"Failed to get task status for {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str, req_logger=get_request_logger("queue_cancel")):
    """Cancel a running or pending task"""

    try:
        task_manager = get_task_manager()
        success = await task_manager.cancel_task(task_id)

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Task cannot be cancelled (not found or already completed)",
            )

        req_logger.info(f"Cancelled task: {task_id}")

        return BaseResponse(success=True, message=f"Task {task_id} has been cancelled")

    except HTTPException:
        raise
    except Exception as e:
        req_logger.error(f"Failed to cancel task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=QueueStatsResponse)
async def list_tasks(
    status: Optional[str] = Query(default=None, description="Filter by status"),
    limit: int = Query(
        default=50, ge=1, le=100, description="Maximum number of tasks to return"
    ),
    req_logger=get_request_logger("queue_list"),
):
    """List tasks with optional status filtering"""

    try:
        task_manager = get_task_manager()

        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = [TaskStatus(status)]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        # Get tasks
        tasks = await task_manager.list_tasks(status_filter=status_filter, limit=limit)

        # Convert to response format
        task_list = []
        for task in tasks:
            task_data = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": (
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                "error_message": task.error_message,
            }
            task_list.append(task_data)

        return QueueStatsResponse(
            success=True,
            message=f"Retrieved {len(task_list)} tasks",
            data={
                "tasks": task_list,
                "total_returned": len(task_list),
                "status_filter": status,
                "limit": limit,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        req_logger.error(f"Failed to list tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=QueueStatsResponse)
async def get_queue_stats(req_logger=get_request_logger("queue_stats")):
    """Get comprehensive queue statistics"""

    try:
        task_manager = get_task_manager()
        stats = await task_manager.get_queue_stats()

        return QueueStatsResponse(
            success=True, message="Queue statistics retrieved", data=stats
        )

    except Exception as e:
        req_logger.error(f"Failed to get queue stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_old_tasks(
    background_tasks: BackgroundTasks, req_logger=get_request_logger("queue_cleanup")
):
    """Clean up old completed tasks"""

    try:
        task_manager = get_task_manager()

        # Run cleanup in background
        background_tasks.add_task(task_manager.cleanup_old_tasks)

        req_logger.info("Started background task cleanup")

        return BaseResponse(success=True, message="Task cleanup started in background")

    except Exception as e:
        req_logger.error(f"Failed to start task cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
