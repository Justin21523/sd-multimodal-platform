"""
Face Restore API Router.

This router provides synchronous face restoration using the postprocess services.
For async processing, use the queue endpoints with task_type="face_restore".
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.postprocess.face_restore_service import get_face_restore_service
from services.history import get_history_store
from utils.image_utils import base64_to_pil_image
from utils.logging_utils import get_request_logger

router = APIRouter(prefix="/face_restore", tags=["Face Restore"])


class FaceRestoreRequest(BaseModel):
    image: str = Field(..., description="Base64 image (data URL or raw base64)")
    model: str = Field(default="GFPGAN_v1.4")
    upscale: int = Field(default=2, ge=1, le=8)
    only_center_face: bool = Field(default=False)
    has_aligned: bool = Field(default=False)
    paste_back: bool = Field(default=True)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    user_id: Optional[str] = Field(default=None)


@router.post("/")
async def restore_faces(payload: FaceRestoreRequest, http_request: Request) -> Dict[str, Any]:
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4())[:8])
    req_logger = get_request_logger(request_id)
    start_time = time.time()

    try:
        service = await get_face_restore_service()
        image = base64_to_pil_image(payload.image)

        result = await service.restore_faces(
            image=image,
            model_name=payload.model,
            upscale=payload.upscale,
            only_center_face=payload.only_center_face,
            has_aligned=payload.has_aligned,
            paste_back=payload.paste_back,
            weight=payload.weight,
            user_id=payload.user_id,
        )

        try:
            store = get_history_store()
            input_params = payload.model_dump(mode="json", exclude_none=True)
            history_id = (
                result.get("task_id")
                if isinstance(result, dict) and isinstance(result.get("task_id"), str)
                else f"face_restore_{int(time.time() * 1000)}"
            )

            inner = result.get("result") if isinstance(result, dict) else None
            if not isinstance(inner, dict):
                inner = {}

            image_path = inner.get("image_path")
            image_url = inner.get("image_url")
            metadata_path = inner.get("metadata_path")
            width = inner.get("restored_width")
            height = inner.get("restored_height")

            try:
                width_int = int(width)  # type: ignore[arg-type]
            except Exception:
                width_int = None
            try:
                height_int = int(height)  # type: ignore[arg-type]
            except Exception:
                height_int = None

            history_result = {
                "success": True,
                "task_id": history_id,
                "task_type": "face_restore",
                "parameters": {
                    "model": payload.model,
                    "upscale": payload.upscale,
                    "only_center_face": payload.only_center_face,
                    "has_aligned": payload.has_aligned,
                    "paste_back": payload.paste_back,
                    "weight": payload.weight,
                },
                "result": {
                    "images": [
                        {
                            "image_path": str(image_path) if image_path else "",
                            "image_url": str(image_url) if image_url else "",
                            "metadata_path": str(metadata_path) if metadata_path else "",
                            "width": width_int,
                            "height": height_int,
                        }
                    ]
                    if image_url or image_path
                    else [],
                    "image_count": 1 if image_url or image_path else 0,
                    "model_used": inner.get("model_used") or payload.model,
                    "processing_time": inner.get("processing_time"),
                    "details": {
                        "faces_detected": inner.get("faces_detected"),
                        "faces_restored": inner.get("faces_restored"),
                    },
                },
            }

            store.record_completion(
                history_id=str(history_id),
                task_type="face_restore",
                run_mode="sync",
                user_id=payload.user_id if isinstance(payload.user_id, str) and payload.user_id else None,
                input_params=input_params,
                result_data=history_result,
            )
        except Exception as e:
            req_logger.warning(f"Failed to write history record: {e}")

        return {
            "success": True,
            "message": "Face restoration completed successfully",
            "data": result,
            "request_id": request_id,
            "timestamp": time.time(),
            "processing_time": round(time.time() - start_time, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        req_logger.error(f"❌ Face restoration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face restoration failed: {e}")


@router.get("/status")
async def get_face_restore_status() -> Dict[str, Any]:
    service = await get_face_restore_service()
    return {"success": True, "data": await service.get_status(), "timestamp": time.time()}
