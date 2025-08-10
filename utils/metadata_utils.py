# utils/metadata_utils.py

"""
SD Multi-Modal Platform - Metadata Management
Phase 1: Metadata Storage and Retrieval
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from .file_utils import ensure_directory


def save_metadata_json(metadata: Dict[str, Any], filepath: Path) -> bool:
    """Save metadata to a JSON file"""
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)

        # 添加儲存時間戳
        metadata["saved_at"] = datetime.now().isoformat()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return True

    except Exception as exc:
        print(f"Failed to save metadata: {exc}")
        return False


def load_metadata_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load metadata from a JSON file"""
    try:
        filepath = Path(filepath)

        if not filepath.exists():
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as exc:
        print(f"Failed to load metadata: {exc}")
        return None


def find_metadata_by_seed(metadata_dir: Path, seed: int) -> List[Path]:
    """Find metadata files by seed value"""

    metadata_dir = Path(metadata_dir)
    matching_files = []

    if not metadata_dir.exists():
        return matching_files

    for filepath in metadata_dir.glob("*.json"):
        try:
            metadata = load_metadata_json(filepath)
            if metadata and metadata.get("seed") == seed:
                matching_files.append(filepath)
        except:
            continue

    return matching_files


def get_recent_generations(metadata_dir: Path, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent generations from metadata directory"""
    metadata_dir = Path(metadata_dir)
    generations = []

    if not metadata_dir.exists():
        return generations

    # Sort metadata files by modification time
    metadata_files = sorted(
        metadata_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    )

    for filepath in metadata_files[:limit]:
        metadata = load_metadata_json(filepath)
        if metadata:
            metadata["metadata_file"] = str(filepath)
            generations.append(metadata)

    return generations
