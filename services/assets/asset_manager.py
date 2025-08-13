# services/assets/asset_manager.py
"""
Comprehensive asset management system for SD Multi-Modal Platform
"""
import json
import time
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import asyncio
from PIL import Image
import uuid

from app.config import settings
from utils.file_utils import safe_filename, ensure_directory
from utils.image_utils import optimize_asset_image, create_thumbnail

logger = logging.getLogger(__name__)


class AssetManager:
    """
    Centralized asset management with metadata tracking,
    duplicate detection, and automatic categorization
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.assets_root = Path(settings.ASSETS_PATH)
        self.metadata_file = self.assets_root / "assets_metadata.json"
        self.thumbnails_dir = self.assets_root / "thumbnails"

        # Asset database (in-memory for Phase 4, can be moved to SQLite later)
        self.assets_db: Dict[str, Dict[str, Any]] = {}
        self.hash_index: Dict[str, str] = {}  # file_hash -> asset_id mapping

        self._initialized = True

    async def initialize(self) -> bool:
        """Initialize asset management system"""
        try:
            logger.info("Initializing Asset Manager...")

            # Create necessary directories
            directories = [
                self.assets_root,
                self.assets_root / "reference",
                self.assets_root / "mask",
                self.assets_root / "pose",
                self.assets_root / "depth",
                self.assets_root / "controlnet",
                self.assets_root / "custom",
                self.thumbnails_dir,
            ]

            for directory in directories:
                ensure_directory(directory)

            # Load existing metadata
            await self._load_metadata()

            # Rebuild hash index
            self._rebuild_hash_index()

            logger.info(
                f"✅ Asset Manager initialized: {len(self.assets_db)} assets loaded"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Asset Manager initialization failed: {str(e)}")
            return False

    async def save_asset(
        self,
        filename: str,
        content: bytes,
        category: str = "reference",
        tags: List[str] = None,  # type: ignore
        description: str = "",
        file_hash: str = None,  # type: ignore
    ) -> Dict[str, Any]:
        """Save asset with metadata and thumbnail generation"""
        try:
            # Generate asset ID and paths
            asset_id = str(uuid.uuid4())
            safe_name = safe_filename(filename)

            # Determine file paths
            category_dir = self.assets_root / category
            file_path = category_dir / f"{asset_id}_{safe_name}"

            # Calculate file hash if not provided
            if not file_hash:
                file_hash = hashlib.sha256(content).hexdigest()

            # Save file
            with open(file_path, "wb") as f:
                f.write(content)

            # Create thumbnail for images
            thumbnail_path = None
            image_info = {}
            try:
                image = Image.open(file_path)
                image_info = {
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                    "mode": image.mode,
                }

                # Create thumbnail
                thumbnail_path = await self._create_thumbnail(image, asset_id)

            except Exception as e:
                logger.warning(f"Failed to process image {filename}: {str(e)}")

            # Prepare asset metadata
            asset_metadata = {
                "asset_id": asset_id,
                "filename": filename,
                "safe_filename": safe_name,
                "file_path": str(file_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
                "category": category,
                "tags": tags or [],
                "description": description,
                "file_hash": file_hash,
                "file_size": len(content),
                "mime_type": mimetypes.guess_type(filename)[0],
                "image_info": image_info,
                "created_at": time.time(),
                "updated_at": time.time(),
                "usage_count": 0,
                "last_used": None,
            }

            # Store in database
            self.assets_db[asset_id] = asset_metadata
            self.hash_index[file_hash] = asset_id

            # Save metadata to disk
            await self._save_metadata()

            logger.info(f"✅ Asset saved: {filename} ({asset_id})")

            return {
                "asset_id": asset_id,
                "file_path": str(file_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
                "metadata": asset_metadata,
            }

        except Exception as e:
            logger.error(f"❌ Failed to save asset {filename}: {str(e)}")
            raise

    async def find_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Find asset by file hash (duplicate detection)"""
        asset_id = self.hash_index.get(file_hash)
        if asset_id:
            return self.assets_db.get(asset_id)
        return None

    async def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get asset metadata by ID"""
        asset = self.assets_db.get(asset_id)
        if asset:
            # Update usage tracking
            asset["usage_count"] += 1
            asset["last_used"] = time.time()
            await self._save_metadata()
        return asset

    async def list_assets(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List assets with filtering and pagination"""
        assets = list(self.assets_db.values())

        # Apply filters
        if category:
            assets = [a for a in assets if a["category"] == category]

        if tags:
            assets = [a for a in assets if any(tag in a["tags"] for tag in tags)]

        # Sort by creation date (newest first)
        assets.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        return assets[offset : offset + limit]

    async def get_categories_with_counts(self) -> Dict[str, int]:
        """Get asset categories with asset counts"""
        categories = {}
        for asset in self.assets_db.values():
            category = asset["category"]
            categories[category] = categories.get(category, 0) + 1
        return categories

    async def delete_asset(self, asset_id: str) -> bool:
        """Delete asset and its files"""
        try:
            asset = self.assets_db.get(asset_id)
            if not asset:
                return False

            # Delete files
            file_path = Path(asset["file_path"])
            if file_path.exists():
                file_path.unlink()

            if asset["thumbnail_path"]:
                thumbnail_path = Path(asset["thumbnail_path"])
                if thumbnail_path.exists():
                    thumbnail_path.unlink()

            # Remove from database
            file_hash = asset["file_hash"]
            del self.assets_db[asset_id]
            if file_hash in self.hash_index:
                del self.hash_index[file_hash]

            # Save metadata
            await self._save_metadata()

            logger.info(f"🗑️ Asset deleted: {asset['filename']} ({asset_id})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete asset {asset_id}: {str(e)}")
            return False

    async def cleanup_orphaned_assets(self) -> Dict[str, Any]:
        """Clean up orphaned files and unused assets"""
        try:
            deleted_files = 0
            freed_space = 0

            # Find orphaned files in asset directories
            for category_dir in self.assets_root.iterdir():
                if not category_dir.is_dir() or category_dir.name == "thumbnails":
                    continue

                for file_path in category_dir.iterdir():
                    if file_path.is_file():
                        # Check if file is referenced in database
                        file_referenced = any(
                            asset["file_path"] == str(file_path)
                            for asset in self.assets_db.values()
                        )

                        if not file_referenced:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            deleted_files += 1
                            freed_space += file_size
                            logger.info(f"🗑️ Deleted orphaned file: {file_path}")

            # Clean up orphaned thumbnails
            for thumbnail_file in self.thumbnails_dir.iterdir():
                if thumbnail_file.is_file():
                    thumbnail_referenced = any(
                        asset["thumbnail_path"] == str(thumbnail_file)
                        for asset in self.assets_db.values()
                    )

                    if not thumbnail_referenced:
                        file_size = thumbnail_file.stat().st_size
                        thumbnail_file.unlink()
                        deleted_files += 1
                        freed_space += file_size
                        logger.info(f"🗑️ Deleted orphaned thumbnail: {thumbnail_file}")

            return {
                "deleted_files": deleted_files,
                "freed_space": f"{freed_space / 1024 / 1024:.2f}MB",
                "freed_space_bytes": freed_space,
            }

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {str(e)}")
            raise

    async def get_asset_stats(self) -> Dict[str, Any]:
        """Get comprehensive asset statistics"""
        total_assets = len(self.assets_db)
        if total_assets == 0:
            return {"total_assets": 0}

        # Calculate statistics
        total_size = sum(asset["file_size"] for asset in self.assets_db.values())
        categories = {}
        tag_counts = {}
        most_used = []

        for asset in self.assets_db.values():
            # Category counts
            category = asset["category"]
            categories[category] = categories.get(category, 0) + 1

            # Tag counts
            for tag in asset["tags"]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Usage tracking
            if asset["usage_count"] > 0:
                most_used.append(
                    {
                        "asset_id": asset["asset_id"],
                        "filename": asset["filename"],
                        "usage_count": asset["usage_count"],
                        "last_used": asset["last_used"],
                    }
                )

        # Sort most used
        most_used.sort(key=lambda x: x["usage_count"], reverse=True)

        return {
            "total_assets": total_assets,
            "total_size": f"{total_size / 1024 / 1024:.2f}MB",
            "total_size_bytes": total_size,
            "categories": categories,
            "top_tags": dict(
                sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "most_used_assets": most_used[:10],
            "average_file_size": f"{total_size / total_assets / 1024:.1f}KB",
        }

    async def _create_thumbnail(
        self, image: Image.Image, asset_id: str
    ) -> Optional[Path]:
        """Create thumbnail for image asset"""
        try:
            # Create thumbnail (256x256 max, maintain aspect ratio)
            thumbnail = image.copy()
            thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)

            # Save thumbnail
            thumbnail_path = self.thumbnails_dir / f"{asset_id}_thumb.jpg"
            thumbnail.save(thumbnail_path, "JPEG", quality=85, optimize=True)

            return thumbnail_path

        except Exception as e:
            logger.warning(f"Failed to create thumbnail for {asset_id}: {str(e)}")
            return None

    async def _load_metadata(self):
        """Load asset metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.assets_db = data.get("assets", {})
                logger.info(f"📄 Loaded {len(self.assets_db)} assets from metadata")
            else:
                logger.info("📄 No existing metadata file found, starting fresh")

        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {str(e)}")
            self.assets_db = {}

    async def _save_metadata(self):
        """Save asset metadata to disk"""
        try:
            metadata = {
                "version": "1.0",
                "last_updated": time.time(),
                "total_assets": len(self.assets_db),
                "assets": self.assets_db,
            }

            # Atomic write using temporary file
            temp_file = self.metadata_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Replace original file
            temp_file.replace(self.metadata_file)

        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {str(e)}")
            raise

    def _rebuild_hash_index(self):
        """Rebuild hash index from asset database"""
        self.hash_index = {
            asset["file_hash"]: asset_id
            for asset_id, asset in self.assets_db.items()
            if "file_hash" in asset
        }
        logger.info(f"🔍 Hash index rebuilt: {len(self.hash_index)} entries")

    async def search_assets(
        self, query: str, category: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search assets by filename, description, or tags"""
        query_lower = query.lower()
        results = []

        for asset in self.assets_db.values():
            # Skip if category filter doesn't match
            if category and asset["category"] != category:
                continue

            # Search in filename, description, and tags
            searchable_text = " ".join(
                [
                    asset["filename"].lower(),
                    asset["description"].lower(),
                    " ".join(asset["tags"]).lower(),
                ]
            )

            if query_lower in searchable_text:
                # Calculate relevance score
                score = 0
                if query_lower in asset["filename"].lower():
                    score += 10
                if query_lower in asset["description"].lower():
                    score += 5
                if any(query_lower in tag.lower() for tag in asset["tags"]):
                    score += 3

                results.append({**asset, "relevance_score": score})

        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:limit]

    async def update_asset_metadata(
        self,
        asset_id: str,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> bool:
        """Update asset metadata"""
        try:
            asset = self.assets_db.get(asset_id)
            if not asset:
                return False

            # Update fields if provided
            if tags is not None:
                asset["tags"] = tags
            if description is not None:
                asset["description"] = description
            if category is not None:
                # Move file if category changed
                if category != asset["category"]:
                    old_path = Path(asset["file_path"])
                    new_dir = self.assets_root / category
                    ensure_directory(new_dir)

                    new_path = new_dir / old_path.name
                    old_path.rename(new_path)
                    asset["file_path"] = str(new_path)
                    asset["category"] = category

            asset["updated_at"] = time.time()

            # Save metadata
            await self._save_metadata()

            logger.info(f"✏️ Asset metadata updated: {asset_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update asset {asset_id}: {str(e)}")
            return False


# Global singleton instance
def get_asset_manager() -> AssetManager:
    """Get global asset manager instance"""
    return AssetManager()
