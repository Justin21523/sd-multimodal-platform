# utils/metadata_utils.py
"""
Metadata management utilities for SD Multi-Modal Platform.
Handles generation metadata, JSON serialization, and metadata search/retrieval.
"""
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from .file_utils import ensure_directory
from utils.logging_utils import setup_logging, get_request_logger

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


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


def find_metadata_by_seed(metadata_dir: Path, seed: int) -> List[Path]:
    """
    Find metadata files by seed value.

    Args:
        metadata_dir: Directory to search for metadata files
        seed: Seed value to search for

    Returns:
        List[Path]: List of metadata files containing the seed
    """

    metadata_dir = Path(metadata_dir)
    matching_files = []

    if not metadata_dir.exists():
        return matching_files

    try:
        # Search for JSON files
        for json_file in metadata_dir.glob("**/*_metadata.json"):
            metadata = load_metadata_json(json_file)
            if metadata and metadata.get("seed") == seed:
                matching_files.append(json_file)

    except Exception as e:
        logger.error(f"Error searching for seed {seed} in {metadata_dir}: {e}")

    return matching_files


def get_recent_generations(metadata_dir: Path, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent generation records sorted by timestamp.

    Args:
        metadata_dir: Directory containing metadata files
        limit: Maximum number of records to return

    Returns:
        List[dict]: List of metadata dictionaries sorted by timestamp (newest first)
    """
    metadata_dir = Path(metadata_dir)
    generations = []

    if not metadata_dir.exists():
        return generations

    try:
        # Collect all metadata files
        for json_file in metadata_dir.glob("**/*_metadata.json"):
            metadata = load_metadata_json(json_file)
            if metadata:
                # Add file path for reference
                metadata["metadata_file"] = str(json_file)
                generations.append(metadata)

        # Sort by timestamp (newest first)
        generations.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        # Limit results
        return generations[:limit]

    except Exception as e:
        logger.error(f"Error getting recent generations from {metadata_dir}: {e}")
        return []


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


def export_metadata_csv(
    metadata_dir: Path, output_file: Path, limit: int = 1000
) -> bool:
    """
    Export metadata to CSV format for analysis.

    Args:
        metadata_dir: Directory containing metadata files
        output_file: Output CSV file path
        limit: Maximum number of records to export

    Returns:
        bool: True if export successful
    """
    try:
        import csv

        metadata_dir = Path(metadata_dir)
        output_file = Path(output_file)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Collect metadata
        records = get_recent_generations(metadata_dir, limit)

        if not records:
            logger.warning("No metadata records found for export")
            return False

        # Define CSV columns
        columns = [
            "timestamp",
            "task_id",
            "model_id",
            "prompt",
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "num_images",
            "generation_time",
            "vram_used_gb",
        ]

        # Write CSV
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            for record in records:
                # Extract only the columns we want
                row = {col: record.get(col, "") for col in columns}
                writer.writerow(row)

        logger.info(f"Exported {len(records)} metadata records to {output_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to export metadata to CSV: {e}")
        return False
