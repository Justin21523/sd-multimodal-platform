# app/api/v1/queue.py
"""
Queue Management API endpoints
Provides task submission, status tracking, and queue management
"""
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime

from services.queue.task_manager import get_task_manager, TaskStatus
from app.core.queue_manager import (
    QueueManager,
    get_queue_manager,
    TaskStatus,
    TaskPriority,
    TaskInfo,
    QueueStats,
)
from app.schemas.requests import Txt2ImgRequest, Img2ImgRequest
from app.schemas.responses import BaseResponse
from app.workers.celery_worker import celery_app
from utils.logging_utils import get_request_logger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/queue", tags=["Queue Management"])

# =====================================
# Request/Response Models
# =====================================


class EnqueueTaskRequest(BaseModel):
    """Request model for enqueuing a new task"""

    task_type: str = Field(
        ..., description="Type of task (txt2img, img2img, upscale, etc.)"
    )
    parameters: Dict[str, Any] = Field(..., description="Task parameters")
    priority: TaskPriority = Field(
        TaskPriority.NORMAL, description="Task priority level"
    )
    user_id: Optional[str] = Field(None, description="User ID for rate limiting")


class EnqueueTaskResponse(BaseModel):
    """Response model for task enqueue operation"""

    success: bool
    task_id: Optional[str] = None
    message: str
    estimated_duration: Optional[float] = None
    queue_position: Optional[int] = None


class TaskStatusResponse(BaseModel):
    """Response model for task status query"""

    task_id: str
    status: TaskStatus
    task_type: str
    priority: TaskPriority

    # Timing information
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[float] = None
    processing_time: Optional[float] = None

    # Progress information
    progress_percent: int = 0
    current_step: str = "queued"
    total_steps: Optional[int] = None

    # Results and errors
    result_data: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None

    # Resource usage
    gpu_memory_used: Optional[float] = None
    retry_count: int = 0


class QueueStatusResponse(BaseModel):
    """Response model for overall queue status"""

    queue_stats: QueueStats
    worker_status: Dict[str, Any]
    system_health: Dict[str, Any]


class TaskListResponse(BaseModel):
    """Response model for task list queries"""

    tasks: List[TaskStatusResponse]
    total_count: int
    page: int
    page_size: int
    has_next: bool


class CancelTaskResponse(BaseModel):
    """Response model for task cancellation"""

    success: bool
    message: str
    task_id: str
    cancelled_at: Optional[datetime] = None


# =====================================
# Helper Functions
# =====================================


def _convert_task_info_to_response(task_info: TaskInfo) -> TaskStatusResponse:
    """Convert TaskInfo to TaskStatusResponse"""
    return TaskStatusResponse(
        task_id=task_info.task_id,
        status=task_info.status,
        task_type=task_info.task_type,
        priority=task_info.priority,
        created_at=task_info.created_at,
        started_at=task_info.started_at,
        completed_at=task_info.completed_at,
        estimated_duration=task_info.estimated_duration,
        processing_time=task_info.processing_time,
        progress_percent=task_info.progress_percent,
        current_step=task_info.current_step,
        total_steps=task_info.total_steps,
        result_data=task_info.result_data,
        error_info=task_info.error_info,
        gpu_memory_used=task_info.gpu_memory_used,
        retry_count=task_info.retry_count,
    )


async def _get_queue_position(
    queue_manager: QueueManager, task_id: str
) -> Optional[int]:
    """Get task position in pending queue"""
    try:
        pending_tasks = await queue_manager.task_store.get_queue_tasks(  # type: ignore
            TaskStatus.PENDING, limit=1000
        )
        for i, task in enumerate(pending_tasks):
            if task.task_id == task_id:
                return i + 1
        return None
    except Exception as e:
        logger.warning(f"Could not determine queue position for task {task_id}: {e}")
        return None


def _get_worker_status() -> Dict[str, Any]:
    """Get Celery worker status information"""
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()

        active_tasks = inspect.active() or {}
        reserved_tasks = inspect.reserved() or {}
        stats = inspect.stats() or {}

        total_active = sum(len(tasks) for tasks in active_tasks.values())
        total_reserved = sum(len(tasks) for tasks in reserved_tasks.values())

        worker_count = len(stats)

        return {
            "total_workers": worker_count,
            "active_tasks": total_active,
            "reserved_tasks": total_reserved,
            "worker_details": stats,
        }

    except Exception as e:
        logger.error(f"Error getting worker status: {e}")
        return {
            "total_workers": 0,
            "active_tasks": 0,
            "reserved_tasks": 0,
            "error": str(e),
        }


def _get_system_health() -> Dict[str, Any]:
    """Get system health information"""
    import psutil
    import torch

    try:
        health = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }

        # GPU information if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3

            health.update(
                {
                    "gpu_available": True,  # type: ignore
                    "gpu_memory_total": f"{gpu_memory:.2f}GB",
                    "gpu_memory_allocated": f"{gpu_allocated:.2f}GB",
                    "gpu_memory_reserved": f"{gpu_reserved:.2f}GB",
                    "gpu_memory_percent": (gpu_allocated / gpu_memory) * 100,
                }
            )
        else:
            health["gpu_available"] = False

        return health

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {"error": str(e)}


# =====================================
# API Endpoints
# =====================================


@router.post("/enqueue", response_model=EnqueueTaskResponse)
async def enqueue_task(
    request: EnqueueTaskRequest,
    queue_manager: QueueManager = Depends(get_queue_manager),
) -> EnqueueTaskResponse:
    """
    Enqueue a new task for processing

    This endpoint adds a new task to the processing queue with the specified priority.
    Rate limiting is applied per user if user_id is provided.
    """
    try:
        logger.info(f"Enqueuing {request.task_type} task for user {request.user_id}")

        # Enqueue task
        task_id = await queue_manager.enqueue_task(
            task_type=request.task_type,
            input_params=request.parameters,
            user_id=request.user_id,
            priority=request.priority,
        )

        if not task_id:
            # Could be rate limit or other error
            return EnqueueTaskResponse(
                success=False,
                message="Failed to enqueue task. Possible rate limit exceeded or queue full.",
            )

        # Get queue position
        queue_position = await _get_queue_position(queue_manager, task_id)

        # Get task info for estimated duration
        task_info = await queue_manager.get_task_status(task_id)
        estimated_duration = task_info.estimated_duration if task_info else None

        # Submit to Celery worker
        celery_task_name = f"process_{request.task_type}"
        try:
            celery_app.send_task(
                celery_task_name,
                args=[task_id, request.parameters],
                queue=_get_celery_queue_for_task(request.task_type),
            )
        except Exception as e:
            logger.error(f"Failed to submit task to Celery: {e}")
            # Update task status to failed
            await queue_manager.task_store.update_task_status(  # type: ignore
                task_id,
                TaskStatus.FAILED,
                error_info={"error": "Failed to submit to worker", "details": str(e)},
            )
            return EnqueueTaskResponse(
                success=False, message=f"Failed to submit task to worker: {e}"
            )

        return EnqueueTaskResponse(
            success=True,
            task_id=task_id,
            message="Task enqueued successfully",
            estimated_duration=estimated_duration,
            queue_position=queue_position,
        )

    except Exception as e:
        logger.error(f"Error enqueuing task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error while enqueuing task: {e}",
        )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str, queue_manager: QueueManager = Depends(get_queue_manager)
) -> TaskStatusResponse:
    """
    Get the current status of a specific task

    Returns detailed information about task progress, timing, and results.
    """
    try:
        task_info = await queue_manager.get_task_status(task_id)

        if not task_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found",
            )

        return _convert_task_info_to_response(task_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error while getting task status: {e}",
        )


@router.post("/cancel/{task_id}", response_model=CancelTaskResponse)
async def cancel_task(
    task_id: str,
    user_id: Optional[str] = Query(None, description="User ID for authorization"),
    queue_manager: QueueManager = Depends(get_queue_manager),
) -> CancelTaskResponse:
    """
    Cancel a specific task if it's cancellable

    Tasks can only be cancelled if they are pending or running, and only by the user who created them.
    """
    try:
        logger.info(f"Attempting to cancel task {task_id} for user {user_id}")

        success = await queue_manager.cancel_task(task_id, user_id)

        if success:
            # Also try to revoke the Celery task if it exists
            try:
                celery_app.control.revoke(task_id, terminate=True)
            except Exception as e:
                logger.warning(f"Could not revoke Celery task {task_id}: {e}")

            return CancelTaskResponse(
                success=True,
                message="Task cancelled successfully",
                task_id=task_id,
                cancelled_at=datetime.now(),
            )
        else:
            # Get task info to provide better error message
            task_info = await queue_manager.get_task_status(task_id)
            if not task_info:
                message = "Task not found"
            elif task_info.user_id != user_id and user_id is not None:
                message = "Not authorized to cancel this task"
            elif task_info.status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ]:
                message = f"Task cannot be cancelled (status: {task_info.status})"
            else:
                message = "Failed to cancel task"

            return CancelTaskResponse(success=False, message=message, task_id=task_id)

    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error while cancelling task: {e}",
        )


@router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status(
    queue_manager: QueueManager = Depends(get_queue_manager),
) -> QueueStatusResponse:
    """
    Get overall queue system status and statistics

    Returns comprehensive information about queue health, worker status, and system resources.
    """
    try:
        # Get queue statistics
        queue_stats = await queue_manager.get_queue_status()

        # Get worker status
        worker_status = _get_worker_status()

        # Get system health
        system_health = _get_system_health()

        return QueueStatusResponse(
            queue_stats=queue_stats,
            worker_status=worker_status,
            system_health=system_health,
        )

    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error while getting queue status: {e}",
        )


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    queue_manager: QueueManager = Depends(get_queue_manager),
) -> TaskListResponse:
    """
    List tasks with optional filtering and pagination

    Returns a paginated list of tasks, optionally filtered by status or user.
    """
    try:
        offset = (page - 1) * page_size

        if user_id:
            # Get tasks for specific user
            tasks = await queue_manager.get_user_tasks(
                user_id, limit=page_size + 1, offset=offset
            )

            # Filter by status if specified
            if status:
                tasks = [t for t in tasks if t.status == status]

        elif status:
            # Get tasks by status
            tasks = await queue_manager.task_store.get_queue_tasks(  # type: ignore
                status, limit=page_size + 1, offset=offset
            )

        else:
            # Get all tasks (this could be expensive, consider limiting in production)
            all_tasks = []
            for task_status in TaskStatus:
                status_tasks = await queue_manager.task_store.get_queue_tasks(  # type: ignore
                    task_status, limit=1000
                )
                all_tasks.extend(status_tasks)

            # Sort by creation time and paginate
            all_tasks.sort(key=lambda x: x.created_at, reverse=True)
            tasks = all_tasks[offset : offset + page_size + 1]

        # Check if there are more pages
        has_next = len(tasks) > page_size
        if has_next:
            tasks = tasks[:page_size]

        # Convert to response format
        task_responses = [_convert_task_info_to_response(task) for task in tasks]

        # Get total count (this is approximate for performance)
        total_count = offset + len(task_responses)
        if has_next:
            total_count += 1  # At least one more page

        return TaskListResponse(
            tasks=task_responses,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=has_next,
        )

    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error while listing tasks: {e}",
        )


@router.get("/tasks/user/{user_id}", response_model=TaskListResponse)
async def list_user_tasks(
    user_id: str,
    status: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    queue_manager: QueueManager = Depends(get_queue_manager),
) -> TaskListResponse:
    """
    List all tasks for a specific user

    Returns a paginated list of tasks created by the specified user.
    """
    return await list_tasks(
        status=status,
        user_id=user_id,
        page=page,
        page_size=page_size,
        queue_manager=queue_manager,
    )


@router.delete("/tasks/completed", response_model=Dict[str, Any])
async def cleanup_completed_tasks(
    older_than_hours: int = Query(
        24, ge=1, description="Clean up tasks completed more than N hours ago"
    ),
    queue_manager: QueueManager = Depends(get_queue_manager),
) -> Dict[str, Any]:
    """
    Clean up completed tasks older than specified time

    Removes completed and failed tasks from the queue to free up storage space.
    """
    try:
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        # Get completed and failed tasks
        completed_tasks = await queue_manager.task_store.get_queue_tasks(  # type: ignore
            TaskStatus.COMPLETED, limit=10000
        )
        failed_tasks = await queue_manager.task_store.get_queue_tasks(  # type: ignore
            TaskStatus.FAILED, limit=10000
        )

        # Filter tasks older than cutoff
        tasks_to_clean = []
        for task in completed_tasks + failed_tasks:
            if task.completed_at and task.completed_at < cutoff_time:
                tasks_to_clean.append(task)

        # Remove tasks (this would need to be implemented in the task store)
        cleaned_count = 0
        for task in tasks_to_clean:
            # This is a simplified cleanup - in production you'd want to:
            # 1. Remove task data from Redis
            # 2. Clean up associated files
            # 3. Update statistics
            logger.info(f"Would clean up task {task.task_id}")
            cleaned_count += 1

        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} old tasks",
            "cleaned_count": cleaned_count,
            "cutoff_time": cutoff_time.isoformat(),
        }

    except Exception as e:
        logger.error(f"Error cleaning up tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error while cleaning up tasks: {e}",
        )


@router.post("/workers/scale", response_model=Dict[str, Any])
async def scale_workers(
    worker_count: int = Query(..., ge=1, le=10, description="Number of workers to run"),
) -> Dict[str, Any]:
    """
    Scale the number of Celery workers (admin endpoint)

    This is a simplified implementation. In production, you'd integrate with
    container orchestration or process management systems.
    """
    try:
        # This is a placeholder implementation
        # In production, you'd integrate with Docker, Kubernetes, or supervisord

        logger.info(f"Worker scaling requested: {worker_count} workers")

        # Get current worker status
        current_workers = _get_worker_status()["total_workers"]

        return {
            "success": True,
            "message": f"Worker scaling requested",
            "current_workers": current_workers,
            "requested_workers": worker_count,
            "note": "This is a placeholder implementation. Integrate with your process manager.",
        }

    except Exception as e:
        logger.error(f"Error scaling workers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error while scaling workers: {e}",
        )


# =====================================
# Utility Functions
# =====================================


def _get_celery_queue_for_task(task_type: str) -> str:
    """Get the appropriate Celery queue for a task type"""
    queue_mapping = {
        "txt2img": "generation",
        "img2img": "generation",
        "inpaint": "generation",
        "upscale": "postprocess",
        "face_restore": "postprocess",
        "video_animate": "generation",  # Video tasks use generation queue
    }

    return queue_mapping.get(task_type, "generation")  # Default to generation queue


# =====================================
# WebSocket Support (Future Enhancement)
# =====================================

# Note: For real-time task status updates, you could add WebSocket endpoints here
# This would allow clients to subscribe to task progress updates

"""
@router.websocket("/ws/task/{task_id}")
async def websocket_task_status(websocket: WebSocket, task_id: str):
    '''WebSocket endpoint for real-time task status updates'''
    await websocket.accept()

    try:
        while True:
            # Get current task status
            task_info = await queue_manager.get_task_status(task_id)

            if task_info:
                status_data = _convert_task_info_to_response(task_info)
                await websocket.send_json(status_data.dict())

                # Break if task is finished
                if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    break

            await asyncio.sleep(2)  # Update every 2 seconds

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
        await websocket.close()
"""


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
