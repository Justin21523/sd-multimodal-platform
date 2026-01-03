"""
Upscale API Router.

This router provides synchronous image upscaling using the postprocess services.
For async processing, use the queue endpoints with task_type="upscale".
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.postprocess.upscale_service import get_upscale_service
from services.history import get_history_store
from utils.image_utils import base64_to_pil_image
from utils.logging_utils import get_request_logger

router = APIRouter(prefix="/upscale", tags=["Upscale"])


class UpscaleRequest(BaseModel):
    image: str = Field(..., description="Base64 image (data URL or raw base64)")
    scale: int = Field(default=4, ge=1, le=8)
    model: str = Field(default="RealESRGAN_x4plus")
    tile_size: Optional[int] = Field(default=None, ge=0, le=2048)
    user_id: Optional[str] = Field(default=None)


@router.post("/")
async def upscale_image(payload: UpscaleRequest, http_request: Request) -> Dict[str, Any]:
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4())[:8])
    req_logger = get_request_logger(request_id)
    start_time = time.time()

    try:
        service = await get_upscale_service()
        image = base64_to_pil_image(payload.image)

        result = await service.upscale_image(
            image=image,
            scale=payload.scale,
            model_name=payload.model,
            tile_size=payload.tile_size,
            user_id=payload.user_id,
        )

        try:
            store = get_history_store()
            input_params = payload.model_dump(mode="json", exclude_none=True)
            history_id = (
                result.get("task_id")
                if isinstance(result, dict) and isinstance(result.get("task_id"), str)
                else f"upscale_{int(time.time() * 1000)}"
            )

            inner = result.get("result") if isinstance(result, dict) else None
            if not isinstance(inner, dict):
                inner = {}

            image_path = inner.get("image_path")
            image_url = inner.get("image_url")
            metadata_path = inner.get("metadata_path")
            width = inner.get("upscaled_width")
            height = inner.get("upscaled_height")

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
                "task_type": "upscale",
                "parameters": {
                    "model": payload.model,
                    "scale": payload.scale,
                    "tile_size": payload.tile_size,
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
                },
            }

            store.record_completion(
                history_id=str(history_id),
                task_type="upscale",
                run_mode="sync",
                user_id=payload.user_id if isinstance(payload.user_id, str) and payload.user_id else None,
                input_params=input_params,
                result_data=history_result,
            )
        except Exception as e:
            req_logger.warning(f"Failed to write history record: {e}")

        return {
            "success": True,
            "message": "Upscale completed successfully",
            "data": result,
            "request_id": request_id,
            "timestamp": time.time(),
            "processing_time": round(time.time() - start_time, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        req_logger.error(f"❌ Upscale failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upscale failed: {e}")


@router.get("/status")
async def get_upscale_status() -> Dict[str, Any]:
    service = await get_upscale_service()
    return {"success": True, "data": await service.get_status(), "timestamp": time.time()}
