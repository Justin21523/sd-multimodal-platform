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
