# utils/metadata_utils.py
"""
Metadata management utilities for SD Multi-Modal Platform.
Extended metadata utilities for generation tracking and analysis
"""
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field
from app.config import settings
from utils.file_utils import ensure_directory, safe_filename

logger = logging.getLogger(__name__)


class ImageMetadata(BaseModel):
    """圖像元數據模型"""

    filename: str
    prompt: str
    negative_prompt: str = ""
    model: str
    width: int
    height: int
    seed: int = -1
    steps: int = 20
    cfg_scale: float = 7.5
    generation_time: float = 0.0
    task_id: str
    task_type: str = "txt2img"
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    # 可擴展的額外參數
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MetadataManager:
    """元數據管理器"""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("./metadata")
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_metadata(self, metadata: ImageMetadata, task_id: str) -> Path:
        """保存元數據"""
        metadata_dir = self.base_path / task_id[:2]  # 使用前兩個字符分組
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = metadata_dir / f"{task_id}.json"

        # 序列化元數據
        metadata_dict = metadata.dict()
        metadata_dict["created_at"] = metadata.created_at.isoformat()

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        return metadata_file

    async def load_metadata(self, metadata_path: Path) -> Optional[ImageMetadata]:
        """載入元數據"""
        try:
            if not metadata_path.exists():
                return None

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

            # 轉換日期字符串
            if "created_at" in metadata_dict:
                metadata_dict["created_at"] = datetime.fromisoformat(
                    metadata_dict["created_at"]
                )

            return ImageMetadata(**metadata_dict)

        except Exception:
            return None

    async def search_metadata(
        self, query: Dict[str, Any], limit: int = 100
    ) -> List[ImageMetadata]:
        """搜索元數據"""
        results = []

        for metadata_file in self.base_path.rglob("*.json"):
            if len(results) >= limit:
                break

            metadata = await self.load_metadata(metadata_file)
            if metadata and self._matches_query(metadata, query):
                results.append(metadata)

        return results

    def _matches_query(self, metadata: ImageMetadata, query: Dict[str, Any]) -> bool:
        """檢查元數據是否符合查詢條件"""
        for key, value in query.items():
            if hasattr(metadata, key):
                attr_value = getattr(metadata, key)
                if isinstance(value, str) and isinstance(attr_value, str):
                    if value.lower() not in attr_value.lower():
                        return False
                elif attr_value != value:
                    return False
            else:
                return False
        return True


async def save_generation_metadata(
    metadata: Dict[str, Any], task_id: str, subfolder: str = "txt2img"
) -> Optional[Path]:
    """
    Save generation metadata to JSON file for reproducibility and analysis
    """
    try:
        # Prepare metadata directory
        metadata_dir = Path(settings.OUTPUT_PATH) / subfolder / "metadata"
        ensure_directory(metadata_dir)

        # Generate metadata filename
        timestamp = int(time.time() * 1000)
        filename = f"{task_id}_{timestamp}_metadata.json"
        metadata_path = metadata_dir / filename

        # Ensure metadata has required fields
        enhanced_metadata = {
            "version": "1.0",
            "task_id": task_id,
            "subfolder": subfolder,
            "saved_at": time.time(),
            **metadata,
        }

        # Add reproducibility information
        if "seed" in metadata:
            enhanced_metadata["reproducibility"] = {
                "seed": metadata["seed"],
                "model_used": metadata.get("model_used", "unknown"),
                "generation_params": metadata.get("generation_params", {}),
                "platform_info": {
                    "device": settings.DEVICE,
                    "torch_dtype": str(settings.get_torch_dtype()),
                    "use_sdpa": settings.USE_SDPA,
                },
            }

        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Saved metadata: {metadata_path}")
        return metadata_path

    except Exception as e:
        logger.error(f"❌ Failed to save metadata: {str(e)}")
        return None


def save_metadata_json(metadata: Dict[str, Any], filepath: Path) -> bool:
    """
    Save generation metadata as JSON file.

    Args:
        metadata: Metadata dictionary to save
        filepath: Path where to save the JSON file

    Returns:
        bool: True if save successful
    """
    try:
        filepath = Path(filepath)
        # Ensure the directory exists
        ensure_directory(filepath.parent)

        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()

        # Write JSON with pretty formatting
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved metadata to {filepath}")
        return True

    except Exception as exc:
        logger.error(f"Failed to save metadata to {filepath}: {exc}")
        return False


def load_metadata_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Load metadata from JSON file.

    Args:
        filepath: Path to JSON metadata file

    Returns:
        dict: Metadata dictionary or None if error
    """
    try:
        filepath = Path(filepath)

        if not filepath.exists():
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        logger.debug(f"Loaded metadata from {filepath}")
        return metadata

    except Exception as exc:
        print(f"Failed to load metadata: {exc}")
        return None


def load_generation_metadata(
    metadata_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Load generation metadata from JSON file"""
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"❌ Failed to load metadata from {metadata_path}: {str(e)}")
        return None


def find_metadata_by_task_id(
    task_id: str, subfolder: Optional[str] = None
) -> Optional[Path]:
    """Find metadata file by task ID"""
    try:
        search_dirs = []

        if subfolder:
            search_dirs.append(Path(settings.OUTPUT_PATH) / subfolder / "metadata")
        else:
            # Search all subfolders
            output_dir = Path(settings.OUTPUT_PATH)
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    metadata_dir = subdir / "metadata"
                    if metadata_dir.exists():
                        search_dirs.append(metadata_dir)

        for metadata_dir in search_dirs:
            for metadata_file in metadata_dir.glob(f"{task_id}_*_metadata.json"):
                return metadata_file

        return None

    except Exception as e:
        logger.error(f"❌ Failed to find metadata for task {task_id}: {str(e)}")
        return None


def find_metadata_by_seed(seed: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Find metadata files by seed value for reproducibility"""
    try:
        matching_metadata = []
        output_dir = Path(settings.OUTPUT_PATH)

        # Search all metadata files
        for metadata_file in output_dir.rglob("*_metadata.json"):
            try:
                metadata = load_generation_metadata(metadata_file)
                if metadata and metadata.get("seed") == seed:
                    metadata["metadata_path"] = str(metadata_file)
                    matching_metadata.append(metadata)

                    if len(matching_metadata) >= limit:
                        break
            except Exception:
                continue

        # Sort by creation time (newest first)
        matching_metadata.sort(key=lambda x: x.get("saved_at", 0), reverse=True)
        return matching_metadata

    except Exception as e:
        logger.error(f"❌ Failed to find metadata by seed {seed}: {str(e)}")
        return []


def get_recent_generations(
    limit: int = 20, subfolder: Optional[str] = None, days: int = 7
) -> List[Dict[str, Any]]:
    """Get recent generation records"""
    try:
        recent_metadata = []
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        search_dirs = []
        if subfolder:
            metadata_dir = Path(settings.OUTPUT_PATH) / subfolder / "metadata"
            if metadata_dir.exists():
                search_dirs.append(metadata_dir)
        else:
            # Search all subfolders
            output_dir = Path(settings.OUTPUT_PATH)
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    metadata_dir = subdir / "metadata"
                    if metadata_dir.exists():
                        search_dirs.append(metadata_dir)

        for metadata_dir in search_dirs:
            for metadata_file in metadata_dir.glob("*_metadata.json"):
                try:
                    metadata = load_generation_metadata(metadata_file)
                    if metadata and metadata.get("saved_at", 0) > cutoff_time:
                        metadata["metadata_path"] = str(metadata_file)
                        recent_metadata.append(metadata)
                except Exception:
                    continue

        # Sort by creation time (newest first) and limit
        recent_metadata.sort(key=lambda x: x.get("saved_at", 0), reverse=True)
        return recent_metadata[:limit]

    except Exception as e:
        logger.error(f"❌ Failed to get recent generations: {str(e)}")
        return []


def get_generation_statistics(days: int = 30) -> Dict[str, Any]:
    """Get generation statistics for the specified period"""
    try:
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        stats = {
            "total_generations": 0,
            "by_subfolder": {},
            "by_model": {},
            "by_day": {},
            "average_processing_time": 0,
            "total_processing_time": 0,
            "most_used_seeds": {},
            "most_common_dimensions": {},
            "period_days": days,
        }

        processing_times = []
        output_dir = Path(settings.OUTPUT_PATH)

        for metadata_file in output_dir.rglob("*_metadata.json"):
            try:
                metadata = load_generation_metadata(metadata_file)
                if not metadata or metadata.get("saved_at", 0) < cutoff_time:
                    continue

                stats["total_generations"] += 1

                # By subfolder
                subfolder = metadata.get("subfolder", "unknown")
                stats["by_subfolder"][subfolder] = (
                    stats["by_subfolder"].get(subfolder, 0) + 1
                )

                # By model
                model = metadata.get("model_used", "unknown")
                stats["by_model"][model] = stats["by_model"].get(model, 0) + 1

                # By day
                day_key = time.strftime(
                    "%Y-%m-%d", time.localtime(metadata.get("saved_at", 0))
                )
                stats["by_day"][day_key] = stats["by_day"].get(day_key, 0) + 1

                # Processing times
                processing_time = metadata.get("processing_time", 0)
                if processing_time > 0:
                    processing_times.append(processing_time)
                    stats["total_processing_time"] += processing_time

                # Seeds
                seed = metadata.get("seed")
                if seed is not None:
                    stats["most_used_seeds"][str(seed)] = (
                        stats["most_used_seeds"].get(str(seed), 0) + 1
                    )

                # Dimensions
                params = metadata.get("generation_params", {})
                width = params.get("width")
                height = params.get("height")
                if width and height:
                    dimension_key = f"{width}x{height}"
                    stats["most_common_dimensions"][dimension_key] = (
                        stats["most_common_dimensions"].get(dimension_key, 0) + 1
                    )

            except Exception:
                continue

        # Calculate averages
        if processing_times:
            stats["average_processing_time"] = sum(processing_times) / len(
                processing_times
            )

        # Sort dictionaries by count
        for key in [
            "by_subfolder",
            "by_model",
            "most_used_seeds",
            "most_common_dimensions",
        ]:
            stats[key] = dict(
                sorted(stats[key].items(), key=lambda x: x[1], reverse=True)
            )

        return stats

    except Exception as e:
        logger.error(f"❌ Failed to get generation statistics: {str(e)}")
        return {"error": str(e)}


def export_metadata_to_csv(
    output_file: Union[str, Path], subfolder: Optional[str] = None, days: int = 30
) -> bool:
    """Export metadata to CSV file for analysis"""
    try:
        import csv

        cutoff_time = time.time() - (days * 24 * 60 * 60)
        metadata_records = []

        # Collect metadata
        search_dirs = []
        if subfolder:
            metadata_dir = Path(settings.OUTPUT_PATH) / subfolder / "metadata"
            if metadata_dir.exists():
                search_dirs.append(metadata_dir)
        else:
            output_dir = Path(settings.OUTPUT_PATH)
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    metadata_dir = subdir / "metadata"
                    if metadata_dir.exists():
                        search_dirs.append(metadata_dir)

        for metadata_dir in search_dirs:
            for metadata_file in metadata_dir.glob("*_metadata.json"):
                try:
                    metadata = load_generation_metadata(metadata_file)
                    if metadata and metadata.get("saved_at", 0) > cutoff_time:
                        # Flatten metadata for CSV
                        flattened = flatten_metadata_for_csv(metadata)
                        metadata_records.append(flattened)
                except Exception:
                    continue

        if not metadata_records:
            logger.warning("No metadata found for CSV export")
            return False

        # Write CSV
        output_path = Path(output_file)
        ensure_directory(output_path.parent)

        # Get all possible field names
        all_fields = set()
        for record in metadata_records:
            all_fields.update(record.keys())

        fieldnames = sorted(all_fields)

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_records)

        logger.info(
            f"📊 Exported {len(metadata_records)} metadata records to {output_path}"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Failed to export metadata to CSV: {str(e)}")
        return False


def flatten_metadata_for_csv(
    metadata: Dict[str, Any], prefix: str = ""
) -> Dict[str, Any]:
    """Flatten nested metadata dictionary for CSV export"""
    flattened = {}

    for key, value in metadata.items():
        new_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            flattened.update(flatten_metadata_for_csv(value, f"{new_key}_"))
        elif isinstance(value, list):
            # Convert lists to string representation
            flattened[new_key] = json.dumps(value) if value else ""
        elif isinstance(value, (str, int, float, bool)):
            flattened[new_key] = value
        else:
            # Convert other types to string
            flattened[new_key] = str(value)

    return flattened


def cleanup_old_metadata(days: int = 30) -> Dict[str, int]:
    """Clean up old metadata files"""
    try:
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        deleted_count = 0
        total_count = 0

        output_dir = Path(settings.OUTPUT_PATH)

        for metadata_file in output_dir.rglob("*_metadata.json"):
            total_count += 1
            try:
                if metadata_file.stat().st_mtime < cutoff_time:
                    metadata_file.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted old metadata: {metadata_file}")
            except Exception as e:
                logger.warning(f"Failed to delete metadata {metadata_file}: {e}")

        return {
            "total_metadata_files": total_count,
            "deleted_files": deleted_count,
            "kept_files": total_count - deleted_count,
        }

    except Exception as e:
        logger.error(f"❌ Failed to cleanup metadata: {str(e)}")
        return {"error": str(e)}  # type: ignore[return-value]


def validate_metadata_integrity() -> Dict[str, Any]:
    """Validate metadata file integrity and consistency"""
    try:
        results = {
            "total_files": 0,
            "valid_files": 0,
            "corrupted_files": [],
            "missing_fields": [],
            "orphaned_metadata": [],
        }

        output_dir = Path(settings.OUTPUT_PATH)

        for metadata_file in output_dir.rglob("*_metadata.json"):
            results["total_files"] += 1

            try:
                metadata = load_generation_metadata(metadata_file)
                if not metadata:
                    results["corrupted_files"].append(str(metadata_file))
                    continue

                # Check required fields
                required_fields = ["task_id", "saved_at", "version"]
                missing = [field for field in required_fields if field not in metadata]
                if missing:
                    results["missing_fields"].append(
                        {"file": str(metadata_file), "missing": missing}
                    )

                # Check if corresponding image files exist
                task_id = metadata.get("task_id")
                subfolder = metadata.get("subfolder", "unknown")
                if task_id:
                    image_dir = output_dir / subfolder
                    image_files = list(image_dir.glob(f"{task_id}_*"))
                    if not image_files:
                        results["orphaned_metadata"].append(str(metadata_file))

                results["valid_files"] += 1

            except Exception as e:
                results["corrupted_files"].append(str(metadata_file))
                logger.warning(f"Corrupted metadata file {metadata_file}: {e}")

        return results

    except Exception as e:
        logger.error(f"❌ Failed to validate metadata integrity: {str(e)}")
        return {"error": str(e)}


def create_generation_report(days: int = 7) -> Dict[str, Any]:
    """Create comprehensive generation report"""
    try:
        stats = get_generation_statistics(days)
        recent_gens = get_recent_generations(limit=10, days=days)
        integrity = validate_metadata_integrity()

        return {
            "report_generated_at": time.time(),
            "period_days": days,
            "statistics": stats,
            "recent_generations": recent_gens,
            "system_integrity": integrity,
            "summary": {
                "total_generations": stats.get("total_generations", 0),
                "avg_processing_time": f"{stats.get('average_processing_time', 0):.2f}s",
                "most_used_model": (
                    list(stats.get("by_model", {}).keys())[0]
                    if stats.get("by_model")
                    else "none"
                ),
                "data_integrity": f"{integrity.get('valid_files', 0)}/{integrity.get('total_files', 0)} files valid",
            },
        }

    except Exception as e:
        logger.error(f"❌ Failed to create generation report: {str(e)}")
        return {"error": str(e)}


def search_metadata_by_prompt(
    metadata_dir: Path, search_term: str, limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Search metadata files by prompt content.

    Args:
        metadata_dir: Directory to search
        search_term: Text to search for in prompts
        limit: Maximum number of results

    Returns:
        List[dict]: Matching metadata records
    """
    metadata_dir = Path(metadata_dir)
    matches = []
    search_term_lower = search_term.lower()

    if not metadata_dir.exists():
        return matches

    try:
        for json_file in metadata_dir.glob("**/*_metadata.json"):
            metadata = load_metadata_json(json_file)
            if metadata:
                prompt = metadata.get("prompt", "").lower()
                if search_term_lower in prompt:
                    metadata["metadata_file"] = str(json_file)
                    matches.append(metadata)

                    if len(matches) >= limit:
                        break

    except Exception as e:
        logger.error(f"Error searching metadata for '{search_term}': {e}")

    return matches


def get_generation_stats(metadata_dir: Path) -> Dict[str, Any]:
    """
    Get statistics about generations from metadata files.

    Args:
        metadata_dir: Directory containing metadata files

    Returns:
        dict: Statistics about generations
    """
    metadata_dir = Path(metadata_dir)
    stats = {
        "total_generations": 0,
        "total_images": 0,
        "models_used": {},
        "average_generation_time": 0.0,
        "date_range": {"earliest": None, "latest": None},
        "common_resolutions": {},
        "total_vram_hours": 0.0,
    }

    if not metadata_dir.exists():
        return stats

    try:
        generation_times = []
        timestamps = []
        total_vram_time = 0.0

        for json_file in metadata_dir.glob("**/*_metadata.json"):
            metadata = load_metadata_json(json_file)
            if metadata:
                stats["total_generations"] += 1

                # Count images
                num_images = metadata.get("num_images", 1)
                stats["total_images"] += num_images

                # Track models used
                model_id = metadata.get("model_id", "unknown")
                stats["models_used"][model_id] = (
                    stats["models_used"].get(model_id, 0) + 1
                )

                # Generation time
                gen_time = metadata.get("generation_time", 0)
                if gen_time > 0:
                    generation_times.append(gen_time)

                # VRAM usage over time
                vram_gb = metadata.get("vram_used_gb", 0)
                if vram_gb > 0 and gen_time > 0:
                    total_vram_time += vram_gb * (gen_time / 3600)  # GB-hours

                # Timestamps
                timestamp = metadata.get("timestamp", 0)
                if timestamp > 0:
                    timestamps.append(timestamp)

                # Resolution tracking
                width = metadata.get("width", 0)
                height = metadata.get("height", 0)
                if width > 0 and height > 0:
                    resolution = f"{width}x{height}"
                    stats["common_resolutions"][resolution] = (
                        stats["common_resolutions"].get(resolution, 0) + 1
                    )

        # Calculate averages and ranges
        if generation_times:
            stats["average_generation_time"] = sum(generation_times) / len(
                generation_times
            )

        if timestamps:
            stats["date_range"]["earliest"] = min(timestamps)
            stats["date_range"]["latest"] = max(timestamps)

        stats["total_vram_hours"] = total_vram_time

    except Exception as e:
        logger.error(f"Error calculating generation stats: {e}")

    return stats


def create_generation_metadata(
    task_id: str,
    request_id: str,
    model_info: Dict[str, Any],
    request_params: Dict[str, Any],
    generation_result: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create standardized generation metadata.

    Args:
        task_id: Unique task identifier
        request_id: Request tracking ID
        model_info: Information about the model used
        request_params: Original request parameters
        generation_result: Results from generation
        additional_data: Optional additional metadata

    Returns:
        dict: Complete metadata record
    """
    metadata = {
        "version": "1.0",
        "timestamp": time.time(),
        "task_id": task_id,
        "request_id": request_id,
        # Model information
        "model_id": model_info.get("model_id"),
        "model_name": model_info.get("model_name"),
        # Generation parameters
        "prompt": request_params.get("prompt"),
        "negative_prompt": request_params.get("negative_prompt", ""),
        "width": request_params.get("width"),
        "height": request_params.get("height"),
        "num_inference_steps": request_params.get("num_inference_steps"),
        "guidance_scale": request_params.get("guidance_scale"),
        "seed": request_params.get("seed"),
        "num_images": request_params.get("num_images", 1),
        # Generation results
        "generation_time": generation_result.get("generation_time"),
        "total_time": generation_result.get("total_time"),
        "vram_used_gb": generation_result.get("vram_used_gb"),
        "vram_delta_gb": generation_result.get("vram_delta_gb"),
        # Optimization info
        "optimization_info": generation_result.get("optimization_info", {}),
        # File information
        "saved_files": generation_result.get("saved_files", []),
        "output_format": "PNG",
    }

    # Add any additional data
    if additional_data:
        metadata.update(additional_data)

    return metadata
