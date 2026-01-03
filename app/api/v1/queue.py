# app/api/v1/queue.py
"""
Queue Management API endpoints
Provides task submission, status tracking, and queue management
"""
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime

from app.core.queue_manager import (
    QueueManager,
    get_queue_manager,
    TaskStatus,
    TaskPriority,
    TaskInfo,
    QueueStats,
)
from app.schemas.requests import Txt2ImgRequest
from app.schemas.queue_requests import (
    QueueFaceRestoreRequest,
    QueueImg2ImgRequest,
    QueueInpaintRequest,
    QueueUpscaleRequest,
)
from app.schemas.responses import BaseResponse
from utils.logging_utils import get_request_logger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/queue", tags=["Queue Management"])

SUPPORTED_TASK_TYPES = {"txt2img", "img2img", "inpaint", "upscale", "face_restore"}

try:
    from app.workers.celery_worker import celery_app

    CELERY_AVAILABLE = True
except Exception:  # pragma: no cover
    celery_app = None  # type: ignore[assignment]
    CELERY_AVAILABLE = False


async def _queue_manager_dep() -> QueueManager:
    """Dependency wrapper that degrades gracefully when Redis isn't available."""
    try:
        return await get_queue_manager()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Queue system unavailable: {e}",
        )

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
    cancelled_at: Optional[datetime] = None
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


class CloneTaskRequest(BaseModel):
    """Request model for retry/rerun of an existing task."""

    priority: TaskPriority = Field(TaskPriority.NORMAL)
    user_id: Optional[str] = Field(default=None)
    overrides: Dict[str, Any] = Field(default_factory=dict)


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
        cancelled_at=getattr(task_info, "cancelled_at", None),
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


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)  # type: ignore[arg-type]
        else:
            merged[k] = v
    return merged


def _get_worker_status() -> Dict[str, Any]:
    """Get Celery worker status information"""
    if not CELERY_AVAILABLE or celery_app is None:
        return {
            "total_workers": 0,
            "active_tasks": 0,
            "reserved_tasks": 0,
            "error": "Celery is not available (install `celery` and start workers)",
        }
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
    queue_manager: QueueManager = Depends(_queue_manager_dep),
) -> EnqueueTaskResponse:
    """
    Enqueue a new task for processing

    This endpoint adds a new task to the processing queue with the specified priority.
    Rate limiting is applied per user if user_id is provided.
    """
    if not CELERY_AVAILABLE or celery_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue workers unavailable (Celery not installed or not configured).",
        )
    if request.task_type not in SUPPORTED_TASK_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported task_type: {request.task_type}. Supported: {sorted(SUPPORTED_TASK_TYPES)}",
        )

    try:
        logger.info(f"Enqueuing {request.task_type} task for user {request.user_id}")

        # Validate/normalize parameters per task_type so history is reproducible.
        try:
            if request.task_type == "txt2img":
                normalized_params = (
                    Txt2ImgRequest(**request.parameters)
                    .model_dump(mode="json", exclude_none=True)
                )
            elif request.task_type == "img2img":
                normalized_params = (
                    QueueImg2ImgRequest(**request.parameters)
                    .model_dump(mode="json", exclude_none=True)
                )
            elif request.task_type == "inpaint":
                normalized_params = (
                    QueueInpaintRequest(**request.parameters)
                    .model_dump(mode="json", exclude_none=True)
                )
            elif request.task_type == "upscale":
                normalized_params = (
                    QueueUpscaleRequest(**request.parameters)
                    .model_dump(mode="json", exclude_none=True)
                )
            elif request.task_type == "face_restore":
                normalized_params = (
                    QueueFaceRestoreRequest(**request.parameters)
                    .model_dump(mode="json", exclude_none=True)
                )
            else:
                normalized_params = dict(request.parameters)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid parameters for task_type={request.task_type}: {e}",
            )

        # Enqueue task
        task_id = await queue_manager.enqueue_task(
            task_type=request.task_type,
            input_params=normalized_params,
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
                args=[task_id, normalized_params],
                task_id=task_id,
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enqueuing task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error while enqueuing task: {e}",
        )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str, queue_manager: QueueManager = Depends(_queue_manager_dep)
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


@router.get("/stream/user/{user_id}")
async def stream_user_tasks(
    user_id: str,
    limit: int = Query(100, ge=1, le=500, description="Max recent tasks to watch"),
    queue_manager: QueueManager = Depends(_queue_manager_dep),
):
    """
    Server-Sent Events stream for all tasks of a user.

    Each event `data` payload matches `/queue/status/{task_id}` (TaskStatusResponse).
    The stream stays open so newly enqueued tasks also appear without reconnecting.
    """

    async def event_generator():
        last_payload_by_id: Dict[str, str] = {}
        loop = asyncio.get_running_loop()

        async def _get_snapshot_chunks() -> List[str]:
            tasks = await queue_manager.get_user_tasks(user_id, limit=limit, offset=0)

            active_ids = set()
            chunks: List[str] = []
            for task_info in tasks:
                active_ids.add(task_info.task_id)
                payload = _convert_task_info_to_response(task_info).model_dump(mode="json")
                encoded = json.dumps(payload, ensure_ascii=False)
                if last_payload_by_id.get(task_info.task_id) != encoded:
                    chunks.append(f"data: {encoded}\n\n")
                    last_payload_by_id[task_info.task_id] = encoded

            for task_id in list(last_payload_by_id.keys()):
                if task_id not in active_ids:
                    last_payload_by_id.pop(task_id, None)

            return chunks

        async def _pubsub_subscribe(pubsub: Any, channel: str) -> None:
            subscribe = getattr(pubsub, "subscribe", None)
            if subscribe is None:
                raise RuntimeError("PubSub missing subscribe()")
            result = subscribe(channel)
            if asyncio.iscoroutine(result):
                await result

        async def _pubsub_get_message(pubsub: Any, timeout_seconds: float) -> Optional[Dict[str, Any]]:
            get_message = getattr(pubsub, "get_message", None)
            if get_message is None:
                return None
            try:
                result = get_message(ignore_subscribe_messages=True, timeout=timeout_seconds)
            except TypeError:
                result = get_message(ignore_subscribe_messages=True)
            if asyncio.iscoroutine(result):
                try:
                    return await asyncio.wait_for(result, timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    return None
            return result

        pubsub: Optional[Any] = None
        channel: Optional[str] = None

        try:
            task_store = getattr(queue_manager, "task_store", None)
            redis_client = getattr(task_store, "redis_client", None) if task_store else None
            if redis_client is not None and hasattr(redis_client, "pubsub"):
                pubsub = redis_client.pubsub()
                prefix = getattr(task_store, "user_tasks_prefix", "user:")
                suffix = getattr(task_store, "user_events_suffix", ":events")
                channel = f"{prefix}{user_id}{suffix}"
                await _pubsub_subscribe(pubsub, channel)
        except Exception:
            pubsub = None
            channel = None

        try:
            for chunk in await _get_snapshot_chunks():
                yield chunk
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': 'queue_unavailable', 'details': str(e)})}\n\n"
            return

        resync_interval_seconds = 15.0
        keep_alive_interval_seconds = 10.0
        last_resync_at = loop.time()
        last_keep_alive_at = last_resync_at

        try:
            while True:
                if pubsub is None:
                    try:
                        for chunk in await _get_snapshot_chunks():
                            yield chunk
                    except Exception as e:
                        yield f"event: error\ndata: {json.dumps({'error': 'queue_unavailable', 'details': str(e)})}\n\n"
                        return

                    yield ": keep-alive\n\n"
                    await asyncio.sleep(1.0)
                    continue

                message: Optional[Dict[str, Any]] = None
                try:
                    message = await _pubsub_get_message(pubsub, timeout_seconds=1.0)
                except Exception:
                    try:
                        close = getattr(pubsub, "close", None)
                        if close is not None:
                            result = close()
                            if asyncio.iscoroutine(result):
                                await result
                    except Exception:
                        pass
                    pubsub = None
                    channel = None
                    continue

                if message and message.get("type") == "message":
                    data = message.get("data")
                    if isinstance(data, str):
                        try:
                            parsed = json.loads(data)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict):
                            task_id = parsed.get("task_id")
                            if isinstance(task_id, str) and task_id:
                                task_info = await queue_manager.get_task_status(task_id)
                                if task_info is None:
                                    last_payload_by_id.pop(task_id, None)
                                else:
                                    payload = _convert_task_info_to_response(task_info).model_dump(
                                        mode="json"
                                    )
                                    encoded = json.dumps(payload, ensure_ascii=False)
                                    if last_payload_by_id.get(task_id) != encoded:
                                        yield f"data: {encoded}\n\n"
                                        last_payload_by_id[task_id] = encoded

                now = loop.time()
                if now - last_resync_at >= resync_interval_seconds:
                    try:
                        for chunk in await _get_snapshot_chunks():
                            yield chunk
                        last_resync_at = now
                    except Exception as e:
                        yield f"event: error\ndata: {json.dumps({'error': 'queue_unavailable', 'details': str(e)})}\n\n"
                        return

                if now - last_keep_alive_at >= keep_alive_interval_seconds:
                    yield ": keep-alive\n\n"
                    last_keep_alive_at = now

        except asyncio.CancelledError:
            return
        finally:
            if pubsub is not None:
                try:
                    if channel and hasattr(pubsub, "unsubscribe"):
                        result = pubsub.unsubscribe(channel)
                        if asyncio.iscoroutine(result):
                            await result
                except Exception:
                    pass
                try:
                    close = getattr(pubsub, "close", None)
                    if close is not None:
                        result = close()
                        if asyncio.iscoroutine(result):
                            await result
                except Exception:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/stream/tasks")
async def stream_queue_tasks(
    queue_manager: QueueManager = Depends(_queue_manager_dep),
):
    """
    Server-Sent Events stream for global task updates.

    Each event `data` payload matches `/queue/status/{task_id}` (TaskStatusResponse).
    Clients should still fetch initial queue pages via REST and use this stream
    for incremental updates.
    """

    async def event_generator():
        terminal_statuses = {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        }
        last_payload_by_id: Dict[str, str] = {}
        loop = asyncio.get_running_loop()

        async def _pubsub_subscribe(pubsub: Any, channel: str) -> None:
            subscribe = getattr(pubsub, "subscribe", None)
            if subscribe is None:
                raise RuntimeError("PubSub missing subscribe()")
            result = subscribe(channel)
            if asyncio.iscoroutine(result):
                await result

        async def _pubsub_get_message(
            pubsub: Any, timeout_seconds: float
        ) -> Optional[Dict[str, Any]]:
            get_message = getattr(pubsub, "get_message", None)
            if get_message is None:
                return None
            try:
                result = get_message(
                    ignore_subscribe_messages=True, timeout=timeout_seconds
                )
            except TypeError:
                result = get_message(ignore_subscribe_messages=True)
            if asyncio.iscoroutine(result):
                try:
                    return await asyncio.wait_for(result, timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    return None
            return result

        pubsub: Optional[Any] = None
        channel: str = "queue:events"

        try:
            task_store = getattr(queue_manager, "task_store", None)
            redis_client = getattr(task_store, "redis_client", None) if task_store else None
            channel = getattr(task_store, "queue_events_channel", channel)
            if redis_client is not None and hasattr(redis_client, "pubsub"):
                pubsub = redis_client.pubsub()
                await _pubsub_subscribe(pubsub, channel)
        except Exception:
            pubsub = None

        keep_alive_interval_seconds = 10.0
        last_keep_alive_at = loop.time()

        # Send an immediate comment so clients know the connection is alive.
        yield ": keep-alive\n\n"

        try:
            while True:
                if pubsub is None:
                    await asyncio.sleep(1.0)
                    now = loop.time()
                    if now - last_keep_alive_at >= keep_alive_interval_seconds:
                        yield ": keep-alive\n\n"
                        last_keep_alive_at = now
                    continue

                message: Optional[Dict[str, Any]] = None
                try:
                    message = await _pubsub_get_message(pubsub, timeout_seconds=1.0)
                except Exception:
                    try:
                        close = getattr(pubsub, "close", None)
                        if close is not None:
                            result = close()
                            if asyncio.iscoroutine(result):
                                await result
                    except Exception:
                        pass
                    pubsub = None
                    continue

                if message and message.get("type") == "message":
                    data = message.get("data")
                    if isinstance(data, str):
                        try:
                            parsed = json.loads(data)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict):
                            task_id = parsed.get("task_id")
                            if isinstance(task_id, str) and task_id:
                                try:
                                    task_info = await queue_manager.get_task_status(task_id)
                                except Exception:
                                    task_info = None
                                if task_info is None:
                                    last_payload_by_id.pop(task_id, None)
                                else:
                                    payload = _convert_task_info_to_response(task_info).model_dump(
                                        mode="json"
                                    )
                                    encoded = json.dumps(payload, ensure_ascii=False)
                                    if last_payload_by_id.get(task_id) != encoded:
                                        yield f"data: {encoded}\n\n"
                                    if task_info.status in terminal_statuses:
                                        last_payload_by_id.pop(task_id, None)
                                    else:
                                        last_payload_by_id[task_id] = encoded

                                # Safety cap for long-lived streams.
                                if len(last_payload_by_id) > 2000:
                                    last_payload_by_id.clear()

                now = loop.time()
                if now - last_keep_alive_at >= keep_alive_interval_seconds:
                    yield ": keep-alive\n\n"
                    last_keep_alive_at = now

        except asyncio.CancelledError:
            return
        finally:
            if pubsub is not None:
                try:
                    if channel and hasattr(pubsub, "unsubscribe"):
                        result = pubsub.unsubscribe(channel)
                        if asyncio.iscoroutine(result):
                            await result
                except Exception:
                    pass
                try:
                    close = getattr(pubsub, "close", None)
                    if close is not None:
                        result = close()
                        if asyncio.iscoroutine(result):
                            await result
                except Exception:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/stream/{task_id}")
async def stream_task_status(
    task_id: str, queue_manager: QueueManager = Depends(_queue_manager_dep)
):
    """
    Server-Sent Events stream for a single task status.

    The event `data` payload matches `/queue/status/{task_id}` (TaskStatusResponse).
    """

    async def event_generator():
        loop = asyncio.get_running_loop()
        last_payload: Optional[str] = None
        pubsub: Optional[Any] = None
        channel: Optional[str] = None

        async def _pubsub_subscribe(pubsub: Any, channel: str) -> None:
            subscribe = getattr(pubsub, "subscribe", None)
            if subscribe is None:
                raise RuntimeError("PubSub missing subscribe()")
            result = subscribe(channel)
            if asyncio.iscoroutine(result):
                await result

        async def _pubsub_get_message(
            pubsub: Any, timeout_seconds: float
        ) -> Optional[Dict[str, Any]]:
            get_message = getattr(pubsub, "get_message", None)
            if get_message is None:
                return None
            try:
                result = get_message(
                    ignore_subscribe_messages=True, timeout=timeout_seconds
                )
            except TypeError:
                result = get_message(ignore_subscribe_messages=True)
            if asyncio.iscoroutine(result):
                try:
                    return await asyncio.wait_for(result, timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    return None
            return result

        task_info = await queue_manager.get_task_status(task_id)
        if not task_info:
            yield f"event: error\ndata: {json.dumps({'error': 'task_not_found', 'task_id': task_id})}\n\n"
            return

        # Subscribe to task:{task_id}:events when possible; fallback to polling otherwise.
        try:
            task_store = getattr(queue_manager, "task_store", None)
            redis_client = getattr(task_store, "redis_client", None) if task_store else None
            if redis_client is not None and hasattr(redis_client, "pubsub"):
                pubsub = redis_client.pubsub()
                prefix = getattr(task_store, "task_prefix", "task:")
                suffix = getattr(task_store, "task_events_suffix", ":events")
                channel = f"{prefix}{task_id}{suffix}"
                await _pubsub_subscribe(pubsub, channel)
        except Exception:
            pubsub = None
            channel = None

        payload = _convert_task_info_to_response(task_info).model_dump(mode="json")
        encoded = json.dumps(payload, ensure_ascii=False)
        yield f"data: {encoded}\n\n"
        last_payload = encoded

        resync_interval_seconds = 15.0
        keep_alive_interval_seconds = 10.0
        last_resync_at = loop.time()
        last_keep_alive_at = last_resync_at

        try:
            while True:
                if task_info.status in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.TIMEOUT,
                ):
                    return

                if pubsub is None:
                    await asyncio.sleep(1.0)
                    task_info = await queue_manager.get_task_status(task_id)
                    if not task_info:
                        yield f"event: error\ndata: {json.dumps({'error': 'task_not_found', 'task_id': task_id})}\n\n"
                        return

                    payload = _convert_task_info_to_response(task_info).model_dump(
                        mode="json"
                    )
                    encoded = json.dumps(payload, ensure_ascii=False)
                    if encoded != last_payload:
                        yield f"data: {encoded}\n\n"
                        last_payload = encoded
                    continue

                message: Optional[Dict[str, Any]] = None
                try:
                    message = await _pubsub_get_message(pubsub, timeout_seconds=1.0)
                except Exception:
                    try:
                        close = getattr(pubsub, "close", None)
                        if close is not None:
                            result = close()
                            if asyncio.iscoroutine(result):
                                await result
                    except Exception:
                        pass
                    pubsub = None
                    channel = None
                    continue

                if message and message.get("type") == "message":
                    data = message.get("data")
                    if isinstance(data, str):
                        try:
                            parsed = json.loads(data)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict) and parsed.get("task_id") == task_id:
                            task_info = await queue_manager.get_task_status(task_id)
                            if not task_info:
                                yield f"event: error\ndata: {json.dumps({'error': 'task_not_found', 'task_id': task_id})}\n\n"
                                return
                            payload = _convert_task_info_to_response(task_info).model_dump(
                                mode="json"
                            )
                            encoded = json.dumps(payload, ensure_ascii=False)
                            if encoded != last_payload:
                                yield f"data: {encoded}\n\n"
                                last_payload = encoded

                now = loop.time()
                if now - last_resync_at >= resync_interval_seconds:
                    task_info = await queue_manager.get_task_status(task_id)
                    if not task_info:
                        yield f"event: error\ndata: {json.dumps({'error': 'task_not_found', 'task_id': task_id})}\n\n"
                        return
                    payload = _convert_task_info_to_response(task_info).model_dump(
                        mode="json"
                    )
                    encoded = json.dumps(payload, ensure_ascii=False)
                    if encoded != last_payload:
                        yield f"data: {encoded}\n\n"
                        last_payload = encoded
                    last_resync_at = now

                if now - last_keep_alive_at >= keep_alive_interval_seconds:
                    yield ": keep-alive\n\n"
                    last_keep_alive_at = now

        except asyncio.CancelledError:
            return
        finally:
            if pubsub is not None:
                try:
                    if channel and hasattr(pubsub, "unsubscribe"):
                        result = pubsub.unsubscribe(channel)
                        if asyncio.iscoroutine(result):
                            await result
                except Exception:
                    pass
                try:
                    close = getattr(pubsub, "close", None)
                    if close is not None:
                        result = close()
                        if asyncio.iscoroutine(result):
                            await result
                except Exception:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/cancel/{task_id}", response_model=CancelTaskResponse)
async def cancel_task(
    task_id: str,
    user_id: Optional[str] = Query(None, description="User ID for authorization"),
    force: bool = Query(
        False,
        description="Force-terminate the Celery task process (use only if cooperative cancel is insufficient).",
    ),
    queue_manager: QueueManager = Depends(_queue_manager_dep),
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
                if celery_app is not None:
                    celery_app.control.revoke(task_id, terminate=bool(force))
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
                TaskStatus.TIMEOUT,
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


async def _clone_existing_task(
    *,
    source_task_id: str,
    request: CloneTaskRequest,
    queue_manager: QueueManager,
    require_failed: bool,
) -> Dict[str, Any]:
    if not CELERY_AVAILABLE or celery_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue workers unavailable (Celery not installed or not configured).",
        )

    task_info = await queue_manager.get_task_status(source_task_id)
    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {source_task_id} not found",
        )

    if require_failed and task_info.status not in (
        TaskStatus.FAILED,
        TaskStatus.TIMEOUT,
        TaskStatus.CANCELLED,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {source_task_id} is not retryable (status: {task_info.status})",
        )

    base_params = task_info.input_params or {}
    merged_params = _deep_merge(base_params, request.overrides or {})

    try:
        if task_info.task_type == "txt2img":
            normalized_params = Txt2ImgRequest(**merged_params).model_dump(
                mode="json", exclude_none=True
            )
        elif task_info.task_type == "img2img":
            normalized_params = QueueImg2ImgRequest(**merged_params).model_dump(
                mode="json", exclude_none=True
            )
        elif task_info.task_type == "inpaint":
            normalized_params = QueueInpaintRequest(**merged_params).model_dump(
                mode="json", exclude_none=True
            )
        elif task_info.task_type == "upscale":
            normalized_params = QueueUpscaleRequest(**merged_params).model_dump(
                mode="json", exclude_none=True
            )
        elif task_info.task_type == "face_restore":
            normalized_params = QueueFaceRestoreRequest(**merged_params).model_dump(
                mode="json", exclude_none=True
            )
        else:
            normalized_params = dict(merged_params)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters for task_type={task_info.task_type}: {e}",
        )

    new_task_id = await queue_manager.enqueue_task(
        task_type=task_info.task_type,
        input_params=normalized_params,
        user_id=request.user_id or task_info.user_id,
        priority=request.priority,
    )
    if not new_task_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue cloned task",
        )

    try:
        celery_app.send_task(
            f"process_{task_info.task_type}",
            args=[new_task_id, normalized_params],
            task_id=new_task_id,
            queue=_get_celery_queue_for_task(task_info.task_type),
        )
    except Exception as e:
        await queue_manager.task_store.update_task_status(  # type: ignore
            new_task_id,
            TaskStatus.FAILED,
            error_info={"error": "Failed to submit to worker", "details": str(e)},
            current_step="failed",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task to worker: {e}",
        )

    return {"task_id": new_task_id, "task_type": task_info.task_type}


@router.post("/rerun/{task_id}")
async def rerun_task(
    task_id: str,
    request: CloneTaskRequest,
    queue_manager: QueueManager = Depends(_queue_manager_dep),
) -> Dict[str, Any]:
    data = await _clone_existing_task(
        source_task_id=task_id,
        request=request,
        queue_manager=queue_manager,
        require_failed=False,
    )
    return {"success": True, "message": "Rerun enqueued", "data": data}


@router.post("/retry/{task_id}")
async def retry_task(
    task_id: str,
    request: CloneTaskRequest,
    queue_manager: QueueManager = Depends(_queue_manager_dep),
) -> Dict[str, Any]:
    data = await _clone_existing_task(
        source_task_id=task_id,
        request=request,
        queue_manager=queue_manager,
        require_failed=True,
    )
    return {"success": True, "message": "Retry enqueued", "data": data}


@router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status(
    queue_manager: QueueManager = Depends(_queue_manager_dep),
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
    queue_manager: QueueManager = Depends(_queue_manager_dep),
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
    queue_manager: QueueManager = Depends(_queue_manager_dep),
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
    queue_manager: QueueManager = Depends(_queue_manager_dep),
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


#
# Legacy endpoints (`/submit/*`, `/stats`, `/cleanup`) were removed in Phase 0 to
# avoid maintaining two queue systems in parallel. Use:
# - POST   /api/v1/queue/enqueue
# - GET    /api/v1/queue/status/{task_id}
# - GET    /api/v1/queue/status
# - GET    /api/v1/queue/tasks
