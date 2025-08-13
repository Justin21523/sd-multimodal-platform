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

from app.schemas.requests import AssetUploadRequest
from services.assets.asset_manager import get_asset_manager
from utils.logging_utils import get_request_logger
from utils.file_utils import safe_filename, ensure_directory
from utils.image_utils import optimize_asset_image, get_image_info
from app.config import settings

router = APIRouter(prefix="/assets", tags=["Asset Management"])
logger = logging.getLogger(__name__)


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
                    uploaded_assets.append(
                        {
                            "filename": file.filename,
                            "asset_id": existing_asset["asset_id"],
                            "status": "duplicate",
                            "path": existing_asset["file_path"],
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

                uploaded_assets.append(
                    {
                        "filename": file.filename,
                        "asset_id": asset_result["asset_id"],
                        "status": "uploaded",
                        "path": asset_result["file_path"],
                        "thumbnail": asset_result.get("thumbnail_path"),
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

        assets = await asset_manager.list_assets(
            category=category, tags=tag_filters, limit=limit, offset=offset
        )

        return {
            "success": True,
            "data": {
                "assets": assets,
                "pagination": {"limit": limit, "offset": offset, "total": len(assets)},
                "filters": {"category": category, "tags": tag_filters},
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

        return {"success": True, "data": asset, "timestamp": time.time()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get asset {asset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get asset: {str(e)}")


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
