"""
Model management API (read-only for Phase 0).

Provides:
- list available models and metadata
- report currently active model (if initialized)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from app.config import settings
from services.models.sd_models import ModelRegistry, get_model_manager

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("")
@router.get("/")
async def list_models() -> Dict[str, Any]:
    manager = get_model_manager()
    current_model_id = getattr(manager, "current_model_id", None)

    models: List[Dict[str, Any]] = []
    for model_id, info in ModelRegistry.AVAILABLE_MODELS.items():
        local_rel = info.get("local_path", "")
        local_abs = str(Path(settings.MODELS_PATH) / local_rel) if local_rel else ""
        models.append(
            {
                "model_id": model_id,
                "name": info.get("name"),
                "type": info.get("type"),
                "capabilities": info.get("capabilities", []),
                "strengths": info.get("strengths", []),
                "recommended_for": info.get("recommended_for", []),
                "vram_requirement": info.get("vram_requirement"),
                "local_path": local_abs,
                "loaded": bool(current_model_id == model_id and manager.is_initialized),
                "active": bool(current_model_id == model_id),
            }
        )

    return {
        "success": True,
        "message": "Models listed",
        "data": {
            "current_model_id": current_model_id,
            "is_initialized": bool(getattr(manager, "is_initialized", False)),
            "models": models,
        },
    }


@router.get("/status")
async def get_model_status() -> Dict[str, Any]:
    manager = get_model_manager()
    status = manager.get_status()
    return {"success": True, "message": "Model status", "data": status}

