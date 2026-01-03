"""
History API (v1).

Provides persistent, queryable run history:
- list/get/export records
- rerun a record via the queue (requires Redis + Celery workers)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.queue_manager import TaskPriority, get_queue_manager
from app.schemas.queue_requests import (
    QueueFaceRestoreRequest,
    QueueImg2ImgRequest,
    QueueInpaintRequest,
    QueueUpscaleRequest,
)
from app.schemas.requests import Txt2ImgRequest
from services.history import get_history_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["History"])

try:
    from app.workers.celery_worker import celery_app

    CELERY_AVAILABLE = True
except Exception:  # pragma: no cover
    celery_app = None  # type: ignore[assignment]
    CELERY_AVAILABLE = False


class HistoryRerunRequest(BaseModel):
    priority: TaskPriority = Field(TaskPriority.NORMAL)
    user_id: Optional[str] = Field(default=None)
    overrides: Dict[str, Any] = Field(default_factory=dict)


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)  # type: ignore[arg-type]
        else:
            merged[k] = v
    return merged


def _get_celery_queue_for_task(task_type: str) -> str:
    mapping = {
        "txt2img": "generation",
        "img2img": "generation",
        "inpaint": "generation",
        "upscale": "postprocess",
        "face_restore": "postprocess",
        "video_animate": "generation",
    }
    return mapping.get(task_type, "generation")


def _normalize_params(task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "txt2img":
        return Txt2ImgRequest(**params).model_dump(mode="json", exclude_none=True)
    if task_type == "img2img":
        return QueueImg2ImgRequest(**params).model_dump(mode="json", exclude_none=True)
    if task_type == "inpaint":
        return QueueInpaintRequest(**params).model_dump(mode="json", exclude_none=True)
    if task_type == "upscale":
        return QueueUpscaleRequest(**params).model_dump(mode="json", exclude_none=True)
    if task_type == "face_restore":
        return QueueFaceRestoreRequest(**params).model_dump(mode="json", exclude_none=True)
    return dict(params)


@router.get("/list")
async def list_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(default=None, description="Search query (prompt/model/task_id/etc.)"),
    task_type: Optional[str] = Query(default=None, description="Filter by task_type"),
    user_id: Optional[str] = Query(default=None, description="Filter by user_id"),
    since: Optional[datetime] = Query(default=None, description="Only records created_at >= since (ISO8601)"),
    until: Optional[datetime] = Query(default=None, description="Only records created_at <= until (ISO8601)"),
) -> Dict[str, Any]:
    store = get_history_store()
    records = store.list_records(
        limit=limit,
        offset=offset,
        q=q,
        task_type=task_type,
        user_id=user_id,
        since=since,
        until=until,
    )
    return {
        "success": True,
        "message": "History listed",
        "data": {"records": records, "limit": limit, "offset": offset, "count": len(records)},
    }


@router.delete("/cleanup")
async def cleanup_history(
    older_than_days: int = Query(30, ge=1, le=3650, description="Delete records older than N days"),
) -> Dict[str, Any]:
    store = get_history_store()
    try:
        result = store.cleanup_records(older_than_days=older_than_days)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return {"success": True, "message": "History cleaned", "data": result}


@router.get("/{history_id}")
async def get_history(history_id: str) -> Dict[str, Any]:
    store = get_history_store()
    record = store.get_record(history_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History record not found")
    return {"success": True, "message": "History record", "data": record}


@router.get("/{history_id}/export")
async def export_history(history_id: str) -> Dict[str, Any]:
    store = get_history_store()
    record = store.get_record(history_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History record not found")
    return {"success": True, "message": "History export", "data": record}


@router.post("/{history_id}/rerun")
async def rerun_history(history_id: str, request: HistoryRerunRequest) -> Dict[str, Any]:
    if not CELERY_AVAILABLE or celery_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue workers unavailable (Celery not installed or not configured).",
        )

    store = get_history_store()
    record = store.get_record(history_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History record not found")

    task_type = record.get("task_type")
    params = record.get("input_params")
    if not isinstance(task_type, str) or not task_type:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid history record (missing task_type)",
        )
    if not isinstance(params, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid history record (missing input_params)",
        )

    merged = _deep_merge(params, request.overrides or {})
    try:
        normalized = _normalize_params(task_type, merged)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters for task_type={task_type}: {e}",
        )

    try:
        queue_manager = await get_queue_manager()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Queue system unavailable: {e}",
        )

    user_id = request.user_id or record.get("user_id")
    user_id = user_id if isinstance(user_id, str) and user_id else None

    task_id = await queue_manager.enqueue_task(  # type: ignore[union-attr]
        task_type=task_type,
        input_params=normalized,
        user_id=user_id,
        priority=request.priority,
    )
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue rerun task",
        )

    celery_app.send_task(
        f"process_{task_type}",
        args=[task_id, normalized],
        task_id=task_id,
        queue=_get_celery_queue_for_task(task_type),
    )

    return {
        "success": True,
        "message": "Rerun enqueued",
        "data": {"task_id": task_id, "task_type": task_type},
    }
