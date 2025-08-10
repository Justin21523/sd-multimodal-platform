# utils/file_utils.py
"""
SD Multi-Modal Platform - File System Utilities
Phase 1: Directory Management and File Operations
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional


def ensure_directory(path: Path) -> Path:
    """Ensure that a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_old_files(directory: Path, days: int) -> int:
    """Delete files older than a specified number of days in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return 0

    cutoff_date = datetime.now() - timedelta(days=days)
    deleted_count = 0

    for file_path in directory.iterdir():
        if file_path.is_file():
            # Check if the file is older than the cutoff date
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)

            if file_time < cutoff_date:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as exc:
                    print(f"Failed to delete {file_path}: {exc}")
                    return 0

    return deleted_count


def get_directory_size(directory: Path) -> int:
    """Calculate the total size of files in a directory."""
    total_size = 0
    directory = Path(directory)

    if directory.exists():
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass

    return total_size


def safe_filename(filename: str) -> str:
    """Sanitize a filename to ensure it is safe for use in the filesystem."""

    import re

    # Replace invalid characters with underscores
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = re.sub(r"\s+", "_", filename)
    filename = filename.strip(".")

    # Limit filename length to 200 characters
    if len(filename) > 200:
        filename = filename[:200]

    return filename


def copy_file_safe(src: Path, dst: Path) -> bool:
    """Safely copy a file from src to dst, ensuring directories exist."""

    try:
        src = Path(src)
        dst = Path(dst)

        # Ensure source file exists
        ensure_directory(dst.parent)

        # Copy the file
        shutil.copy2(src, dst)
        return True

    except Exception as exc:
        print(f"File copy failed: {exc}")
        return False
