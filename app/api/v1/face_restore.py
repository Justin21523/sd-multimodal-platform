"""
Face Restore API Router.

This router provides synchronous face restoration using the postprocess services.
For async processing, use the queue endpoints with task_type="face_restore".
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import settings
from app.schemas.requests import ControlNetConfig
from services.assets.asset_manager import get_asset_manager
from services.postprocess.face_restore_service import get_face_restore_service
from services.history import get_history_store
from utils.image_utils import base64_to_pil_image
from utils.logging_utils import get_request_logger

router = APIRouter(prefix="/face_restore", tags=["Face Restore"])


class FaceRestoreRequest(BaseModel):
    image: Optional[str] = Field(
        default=None, description="Base64 image (data URL or raw base64) (legacy)"
    )
    image_asset_id: Optional[str] = Field(
        default=None, description="Asset ID for input image (preferred)"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path to input image (restricted to ASSETS_PATH/OUTPUT_PATH)",
    )
    model: str = Field(default="GFPGAN_v1.4")
    upscale: int = Field(default=2, ge=1, le=8)
    only_center_face: bool = Field(default=False)
    has_aligned: bool = Field(default=False)
    paste_back: bool = Field(default=True)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    user_id: Optional[str] = Field(default=None)

    @field_validator("image")
    @classmethod
    def validate_image_base64(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return ControlNetConfig.validate_base64_image(v)

    @field_validator("image_asset_id")
    @classmethod
    def validate_image_asset_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        try:
            uuid.UUID(str(v))
        except Exception as e:
            raise ValueError(f"Invalid image_asset_id (expected UUID): {e}")
        return str(v)

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        from app.schemas.queue_requests import _resolve_allowed_image_path

        return _resolve_allowed_image_path(v)

    @model_validator(mode="after")
    def validate_source(self) -> "FaceRestoreRequest":
        provided = [
            self.image is not None,
            self.image_asset_id is not None,
            self.image_path is not None,
        ]
        if sum(provided) != 1:
            raise ValueError(
                "face_restore requires exactly one of: image (base64), image_asset_id, image_path"
            )
        return self


@router.post("/")
async def restore_faces(payload: FaceRestoreRequest, http_request: Request) -> Dict[str, Any]:
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4())[:8])
    req_logger = get_request_logger(request_id)
    start_time = time.time()

    try:
        service = await get_face_restore_service()
        image_input: Any
        if payload.image is not None:
            image_input = base64_to_pil_image(payload.image)
        elif payload.image_asset_id is not None:
            asset_manager = get_asset_manager()
            asset = await asset_manager.get_asset(payload.image_asset_id)
            if not asset:
                raise HTTPException(status_code=404, detail="Asset not found")
            file_path = asset.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                raise HTTPException(status_code=404, detail="Asset missing file_path")
            resolved = Path(file_path).expanduser().resolve()
            assets_root = Path(settings.ASSETS_PATH).expanduser().resolve()
            try:
                if not resolved.is_relative_to(assets_root):
                    raise HTTPException(
                        status_code=422,
                        detail="image_asset_id must resolve under ASSETS_PATH (security restriction)",
                    )
            except AttributeError:  # pragma: no cover (py<3.9)
                if not str(resolved).startswith(str(assets_root) + "/"):
                    raise HTTPException(
                        status_code=422,
                        detail="image_asset_id must resolve under ASSETS_PATH (security restriction)",
                    )
            if not resolved.exists() or not resolved.is_file():
                raise HTTPException(status_code=404, detail="Asset file not found on disk")
            image_input = resolved
        else:
            assert payload.image_path is not None
            image_input = Path(payload.image_path)

        result = await service.restore_faces(
            image=image_input,
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
