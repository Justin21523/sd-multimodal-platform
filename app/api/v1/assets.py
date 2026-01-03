# app/api/v1/assets.py
"""
Asset management API for reference images, masks, poses, and custom assets
"""
from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List, Optional
import logging
import time
import hashlib
import mimetypes
from pathlib import Path
import json
import shutil
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.schemas.requests import AssetUploadRequest
from services.assets.asset_manager import get_asset_manager
from services.history import get_history_store
from utils.logging_utils import get_request_logger
from utils.file_utils import safe_filename, ensure_directory
from utils.image_utils import optimize_asset_image, get_image_info
from app.config import settings

router = APIRouter(prefix="/assets", tags=["Asset Management"])
logger = logging.getLogger(__name__)


def _decorate_asset(asset: Dict[str, Any]) -> Dict[str, Any]:
    """Add URL/relative-path fields for UI consumption."""
    decorated = dict(asset)
    asset_id = decorated.get("asset_id")
    decorated["download_url"] = (
        f"{settings.API_PREFIX}/assets/{asset_id}/download" if asset_id else None
    )

    assets_root = Path(settings.ASSETS_PATH)

    def _path_to_url(path_value: Optional[str]) -> Optional[str]:
        if not path_value:
            return None
        try:
            rel = Path(path_value).relative_to(assets_root)
            return f"/assets/{rel.as_posix()}"
        except Exception:
            return None

    decorated["file_url"] = _path_to_url(decorated.get("file_path"))
    decorated["thumbnail_url"] = _path_to_url(decorated.get("thumbnail_path"))
    return decorated


def _resolve_output_file_from_url(raw_url: str) -> Path:
    raw_url = raw_url.strip()
    if not raw_url:
        raise HTTPException(status_code=422, detail="image_url is required")

    parsed = urlparse(raw_url)
    path = parsed.path or raw_url
    if path.startswith("/outputs/"):
        rel = path[len("/outputs/") :]
    elif path.startswith("outputs/"):
        rel = path[len("outputs/") :]
    else:
        raise HTTPException(
            status_code=422,
            detail="image_url must be under /outputs (security restriction)",
        )

    output_root = Path(str(settings.OUTPUT_PATH)).expanduser().resolve()
    src_path = (output_root / rel).resolve()
    try:
        if not src_path.is_relative_to(output_root):
            raise HTTPException(
                status_code=422,
                detail="Resolved path is outside OUTPUT_PATH (security restriction)",
            )
    except AttributeError:  # pragma: no cover (py<3.9)
        if not str(src_path).startswith(str(output_root) + "/"):
            raise HTTPException(
                status_code=422,
                detail="Resolved path is outside OUTPUT_PATH (security restriction)",
            )

    if not src_path.exists() or not src_path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found on disk")
    return src_path


class AssetUpdateRequest(BaseModel):
    tags: Optional[List[str]] = Field(default=None, description="Replace tags list")
    description: Optional[str] = Field(default=None, description="Replace description")
    category: Optional[str] = Field(default=None, description="Move asset to category")


class AssetImportFromOutputRequest(BaseModel):
    """Server-side import from /outputs (no download+reupload)."""

    image_url: str = Field(..., description="URL/path under /outputs to import")
    category: str = Field(default="reference")
    tags: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    filename: Optional[str] = Field(default=None, description="Override stored filename")


class AssetImportFromHistoryRequest(BaseModel):
    history_id: str = Field(..., description="History record ID")
    category: str = Field(default="reference")
    tags: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    max_images: int = Field(default=50, ge=1, le=200)
    image_indexes: Optional[List[int]] = Field(
        default=None, description="Optional output image indexes to import"
    )


class AssetBatchDeleteRequest(BaseModel):
    asset_ids: List[str] = Field(..., min_length=1, description="Asset IDs to delete")


class AssetBatchUpdateItem(BaseModel):
    asset_id: str
    tags: Optional[List[str]] = Field(default=None)
    description: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)


class AssetBatchUpdateRequest(BaseModel):
    items: List[AssetBatchUpdateItem] = Field(..., min_length=1)


@router.post("/upload")
async def upload_assets(
    files: List[UploadFile] = File(...),
    category: str = Form(default="reference"),
    tags: str = Form(default=""),
    descriptions: str = Form(default=""),
    http_request: Request = None,  # type: ignore
    background_tasks: BackgroundTasks = None,  # type: ignore
) -> Dict[str, Any]:
    """
    Upload multiple assets with metadata

    Supports:
    - Multiple file upload with drag & drop
    - Automatic file type detection and validation
    - Duplicate detection via file hashing
    - Asset categorization and tagging
    - Thumbnail generation for large images
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    req_logger = get_request_logger(request_id)

    start_time = time.time()

    try:
        req_logger.info(
            f"📁 Starting asset upload: {len(files)} files",
            extra={"category": category, "file_count": len(files)},
        )

        # Parse optional metadata
        tag_list = (
            [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        )
        desc_list = (
            [desc.strip() for desc in descriptions.split("|") if desc.strip()]
            if descriptions
            else []
        )

        # Validate category
        valid_categories = [
            "reference",
            "mask",
            "pose",
            "depth",
            "controlnet",
            "custom",
        ]
        if category not in valid_categories:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid category. Must be one of: {valid_categories}",
            )

        # Process uploads
        asset_manager = get_asset_manager()
        uploaded_assets = []
        failed_uploads = []

        for i, file in enumerate(files):
            try:
                # Validate file
                if not file.filename:
                    failed_uploads.append(
                        {"filename": "unknown", "error": "No filename provided"}
                    )
                    continue

                # Check file size (10MB limit)
                file_size = 0
                content = await file.read()
                file_size = len(content)

                if file_size > 10 * 1024 * 1024:  # 10MB
                    failed_uploads.append(
                        {
                            "filename": file.filename,
                            "error": f"File too large: {file_size / 1024 / 1024:.1f}MB > 10MB",
                        }
                    )
                    continue

                # Validate MIME type
                mime_type, _ = mimetypes.guess_type(file.filename)
                if not mime_type or not mime_type.startswith("image/"):
                    failed_uploads.append(
                        {
                            "filename": file.filename,
                            "error": f"Invalid file type: {mime_type}",
                        }
                    )
                    continue

                # Generate file hash for duplicate detection
                file_hash = hashlib.sha256(content).hexdigest()

                # Check for duplicates
                existing_asset = await asset_manager.find_by_hash(file_hash)
                if existing_asset:
                    req_logger.info(f"Duplicate file detected: {file.filename}")
                    decorated = _decorate_asset(existing_asset)
                    uploaded_assets.append(
                        {
                            "filename": file.filename,
                            "asset_id": existing_asset["asset_id"],
                            "status": "duplicate",
                            "path": existing_asset["file_path"],
                            "file_url": decorated.get("file_url"),
                            "thumbnail_url": decorated.get("thumbnail_url"),
                            "download_url": decorated.get("download_url"),
                        }
                    )
                    continue

                # Save asset
                asset_result = await asset_manager.save_asset(
                    filename=file.filename,
                    content=content,
                    category=category,
                    tags=tag_list,
                    description=desc_list[i] if i < len(desc_list) else "",
                    file_hash=file_hash,
                )

                decorated_meta = _decorate_asset(asset_result.get("metadata", {}))
                uploaded_assets.append(
                    {
                        "filename": file.filename,
                        "asset_id": asset_result["asset_id"],
                        "status": "uploaded",
                        "path": asset_result["file_path"],
                        "thumbnail": asset_result.get("thumbnail_path"),
                        "file_url": decorated_meta.get("file_url"),
                        "thumbnail_url": decorated_meta.get("thumbnail_url"),
                        "download_url": decorated_meta.get("download_url"),
                        "file_size": file_size,
                        "hash": file_hash,
                    }
                )

                req_logger.info(f"✅ Asset uploaded: {file.filename}")

            except Exception as e:
                failed_uploads.append({"filename": file.filename, "error": str(e)})
                req_logger.error(f"❌ Failed to upload {file.filename}: {str(e)}")

        processing_time = time.time() - start_time

        return {
            "success": True,
            "message": f"Asset upload completed: {len(uploaded_assets)} successful, {len(failed_uploads)} failed",
            "data": {
                "uploaded_assets": uploaded_assets,
                "failed_uploads": failed_uploads,
                "summary": {
                    "total_files": len(files),
                    "successful": len(uploaded_assets),
                    "failed": len(failed_uploads),
                    "duplicates": len(
                        [a for a in uploaded_assets if a["status"] == "duplicate"]
                    ),
                },
                "processing_time": processing_time,
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        req_logger.error(f"❌ Asset upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Asset upload failed: {str(e)}")


@router.get("/list")
async def list_assets(
    category: Optional[str] = None,
    tags: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List assets with filtering and pagination"""
    try:
        asset_manager = get_asset_manager()

        # Parse filters
        tag_filters = (
            [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else None
        )

        # NOTE: AssetManager is in-memory for now; keep pagination correct by filtering first.
        all_assets = await asset_manager.list_assets(
            category=category, tags=tag_filters, limit=1_000_000, offset=0
        )

        if q:
            q_lower = q.lower()
            filtered = []
            for asset in all_assets:
                haystack = " ".join(
                    [
                        str(asset.get("filename", "")).lower(),
                        str(asset.get("description", "")).lower(),
                        " ".join(asset.get("tags") or []).lower(),
                    ]
                )
                if q_lower in haystack:
                    filtered.append(asset)
            all_assets = filtered

        total = len(all_assets)
        assets = all_assets[offset : offset + limit]

        return {
            "success": True,
            "data": {
                "assets": [_decorate_asset(a) for a in assets],
                "pagination": {"limit": limit, "offset": offset, "total": total},
                "filters": {"category": category, "tags": tag_filters, "q": q},
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"❌ Failed to list assets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list assets: {str(e)}")


@router.get("/categories")
async def get_asset_categories() -> Dict[str, Any]:
    """Get available asset categories with counts"""
    try:
        asset_manager = get_asset_manager()
        categories = await asset_manager.get_categories_with_counts()

        return {
            "success": True,
            "data": {
                "categories": categories,
                "total_assets": sum(categories.values()),
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"❌ Failed to get categories: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get categories: {str(e)}"
        )


@router.get("/{asset_id}")
async def get_asset(asset_id: str) -> Dict[str, Any]:
    """Get specific asset metadata and info"""
    try:
        asset_manager = get_asset_manager()
        asset = await asset_manager.get_asset(asset_id)

        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")

        return {"success": True, "data": _decorate_asset(asset), "timestamp": time.time()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get asset {asset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get asset: {str(e)}")


@router.patch("/{asset_id}")
async def update_asset(asset_id: str, payload: AssetUpdateRequest) -> Dict[str, Any]:
    """Update asset metadata (tags/description/category)."""
    try:
        asset_manager = get_asset_manager()

        valid_categories = [
            "reference",
            "mask",
            "pose",
            "depth",
            "controlnet",
            "custom",
        ]
        if payload.category and payload.category not in valid_categories:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid category. Must be one of: {valid_categories}",
            )

        tags = payload.tags
        if tags is not None:
            tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]

        ok = await asset_manager.update_asset_metadata(
            asset_id=asset_id,
            tags=tags,
            description=payload.description,
            category=payload.category,
        )
        if not ok:
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")

        asset = await asset_manager.get_asset(asset_id)
        return {"success": True, "data": _decorate_asset(asset), "timestamp": time.time()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update asset {asset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update asset: {str(e)}")


@router.get("/{asset_id}/download")
async def download_asset(asset_id: str):
    """Download asset file"""
    try:
        asset_manager = get_asset_manager()
        asset = await asset_manager.get_asset(asset_id)

        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")

        file_path = Path(asset["file_path"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Asset file not found on disk")

        return FileResponse(
            path=str(file_path),
            filename=asset["filename"],
            media_type=asset.get("mime_type", "application/octet-stream"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to download asset {asset_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to download asset: {str(e)}"
        )


@router.delete("/{asset_id}")
async def delete_asset(asset_id: str, http_request: Request) -> Dict[str, Any]:
    """Delete an asset and its files"""
    request_id = getattr(http_request.state, "request_id", "unknown")
    req_logger = get_request_logger(request_id)

    try:
        asset_manager = get_asset_manager()

        # Get asset info before deletion
        asset = await asset_manager.get_asset(asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")

        # Delete asset
        success = await asset_manager.delete_asset(asset_id)

        if success:
            req_logger.info(f"🗑️ Asset deleted: {asset_id}")
            return {
                "success": True,
                "message": f"Asset deleted successfully: {asset['filename']}",
                "data": {"deleted_asset": asset_id, "filename": asset["filename"]},
                "timestamp": time.time(),
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete asset")

    except HTTPException:
        raise
    except Exception as e:
        req_logger.error(f"❌ Failed to delete asset {asset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete asset: {str(e)}")


@router.post("/import_from_output")
async def import_asset_from_output(payload: AssetImportFromOutputRequest) -> Dict[str, Any]:
    """
    Import an existing file under `/outputs` into the asset library (server-side copy).

    This avoids downloading the image to the browser and uploading it again.
    """
    try:
        asset_manager = get_asset_manager()

        valid_categories = [
            "reference",
            "mask",
            "pose",
            "depth",
            "controlnet",
            "custom",
        ]
        if payload.category not in valid_categories:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid category. Must be one of: {valid_categories}",
            )

        src_path = _resolve_output_file_from_url(payload.image_url)

        filename = payload.filename or src_path.name
        content = src_path.read_bytes()
        tags = [t.strip() for t in payload.tags if isinstance(t, str) and t.strip()]

        result = await asset_manager.save_asset(
            filename=filename,
            content=content,
            category=payload.category,
            tags=tags,
            description=payload.description or "",
        )

        decorated = _decorate_asset(result.get("metadata", {}))
        return {
            "success": True,
            "message": "Imported from outputs",
            "data": {
                "asset_id": result.get("asset_id"),
                "metadata": decorated,
            },
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to import asset from output: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to import asset: {str(e)}")


@router.post("/import_from_history")
async def import_assets_from_history(payload: AssetImportFromHistoryRequest) -> Dict[str, Any]:
    """
    Batch import output images from a History v1 record into the asset library.

    This is server-side only and enforces that sources are under `/outputs`.
    """
    try:
        valid_categories = [
            "reference",
            "mask",
            "pose",
            "depth",
            "controlnet",
            "custom",
        ]
        if payload.category not in valid_categories:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid category. Must be one of: {valid_categories}",
            )

        store = get_history_store()
        rec = store.get_record(payload.history_id)
        if not rec:
            raise HTTPException(status_code=404, detail="History record not found")

        output_images = rec.get("output_images")
        if not isinstance(output_images, list) or not output_images:
            raise HTTPException(status_code=400, detail="History record has no output_images")

        indexes = payload.image_indexes
        if indexes is not None:
            if not isinstance(indexes, list) or not indexes:
                raise HTTPException(status_code=422, detail="image_indexes must be a non-empty list")
            selected = []
            for i in indexes:
                if not isinstance(i, int) or i < 0 or i >= len(output_images):
                    raise HTTPException(status_code=422, detail=f"Invalid image index: {i}")
                selected.append(output_images[i])
            output_images = selected

        output_images = output_images[: payload.max_images]

        asset_manager = get_asset_manager()
        tags = [t.strip() for t in payload.tags if isinstance(t, str) and t.strip()]

        imported: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for idx, img in enumerate(output_images):
            if not isinstance(img, dict):
                failed.append({"index": idx, "error": "Invalid output image entry"})
                continue
            image_url = img.get("image_url") or img.get("url")
            if not isinstance(image_url, str) or not image_url.strip():
                failed.append({"index": idx, "error": "Missing image_url"})
                continue

            try:
                src_path = _resolve_output_file_from_url(image_url)
                result = await asset_manager.save_asset(
                    filename=src_path.name,
                    content=src_path.read_bytes(),
                    category=payload.category,
                    tags=tags,
                    description=payload.description or "",
                )
                decorated = _decorate_asset(result.get("metadata", {}))
                imported.append(
                    {
                        "index": idx,
                        "source_image_url": image_url,
                        "asset_id": result.get("asset_id"),
                        "metadata": decorated,
                    }
                )
            except HTTPException as e:
                failed.append({"index": idx, "source_image_url": image_url, "error": str(e.detail)})
            except Exception as e:
                failed.append({"index": idx, "source_image_url": image_url, "error": str(e)})

        return {
            "success": True,
            "message": "Imported from history",
            "data": {
                "history_id": payload.history_id,
                "imported": imported,
                "failed": failed,
            },
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to import assets from history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to import from history: {str(e)}")


@router.post("/batch/delete")
async def batch_delete_assets(payload: AssetBatchDeleteRequest) -> Dict[str, Any]:
    """Delete multiple assets in one request."""
    asset_manager = get_asset_manager()
    deleted: List[str] = []
    failed: List[Dict[str, Any]] = []
    for asset_id in payload.asset_ids:
        try:
            ok = await asset_manager.delete_asset(asset_id)
            if ok:
                deleted.append(asset_id)
            else:
                failed.append({"asset_id": asset_id, "error": "not_found"})
        except Exception as e:
            failed.append({"asset_id": asset_id, "error": str(e)})
    return {
        "success": True,
        "message": "Batch delete completed",
        "data": {"deleted": deleted, "failed": failed},
        "timestamp": time.time(),
    }


@router.post("/batch/update")
async def batch_update_assets(payload: AssetBatchUpdateRequest) -> Dict[str, Any]:
    """Update multiple assets in one request."""
    asset_manager = get_asset_manager()
    updated: List[str] = []
    failed: List[Dict[str, Any]] = []
    for item in payload.items:
        try:
            tags = item.tags
            if tags is not None:
                tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
            ok = await asset_manager.update_asset_metadata(
                asset_id=item.asset_id,
                tags=tags,
                description=item.description,
                category=item.category,
            )
            if ok:
                updated.append(item.asset_id)
            else:
                failed.append({"asset_id": item.asset_id, "error": "not_found"})
        except Exception as e:
            failed.append({"asset_id": item.asset_id, "error": str(e)})
    return {
        "success": True,
        "message": "Batch update completed",
        "data": {"updated": updated, "failed": failed},
        "timestamp": time.time(),
    }


@router.post("/cleanup")
async def cleanup_orphaned_assets(http_request: Request) -> Dict[str, Any]:
    """Clean up orphaned assets and unused files"""
    request_id = getattr(http_request.state, "request_id", "unknown")
    req_logger = get_request_logger(request_id)

    try:
        asset_manager = get_asset_manager()
        cleanup_result = await asset_manager.cleanup_orphaned_assets()

        req_logger.info(
            f"🧹 Asset cleanup completed",
            extra={
                "deleted_files": cleanup_result["deleted_files"],
                "freed_space": cleanup_result["freed_space"],
            },
        )

        return {
            "success": True,
            "message": "Asset cleanup completed",
            "data": cleanup_result,
            "timestamp": time.time(),
        }

    except Exception as e:
        req_logger.error(f"❌ Asset cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Asset cleanup failed: {str(e)}")
