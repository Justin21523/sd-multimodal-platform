# utils/file_utils.py
"""
File system utilities for SD Multi-Modal Platform.
Handles file operations, directory management, and safe filename generation.
"""

import os
import re
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from utils.logging_utils import setup_logging, get_request_logger

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create

    Returns:
        Path: The created/existing directory path
    """
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def safe_filename(filename: str, max_length: int = 50) -> str:
    """
    Generate a safe filename by removing/replacing special characters.

    Args:
        filename: Original filename
        max_length: Maximum length of resulting filename

    Returns:
        str: Safe filename
    """
    # Remove or replace unsafe characters
    safe_name = re.sub(r"[^\w\s-]", "", filename)
    safe_name = re.sub(r"[-\s]+", "-", safe_name)
    safe_name = safe_name.strip("-")

    # Truncate if too long
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]

    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed"

    return safe_name


def cleanup_old_files(directory: Path, days: int) -> int:
    """
    Remove files older than specified days.

    Args:
        directory: Directory to clean
        days: Files older than this many days will be removed

    Returns:
        int: Number of files removed
    """
    if days <= 0:
        return 0

    directory = Path(directory)
    if not directory.exists():
        return 0

    cutoff_time = time.time() - (days * 24 * 3600)
    removed_count = 0

    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old files from {directory}")

    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")

    return removed_count


def get_directory_size(directory: Path) -> int:
    """
    Calculate total directory size in bytes.

    Args:
        directory: Directory to measure

    Returns:
        int: Total size in bytes
    """
    total_size = 0
    directory = Path(directory)

    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except Exception as e:
        logger.error(f"Error calculating directory size for {directory}: {e}")

    return total_size


def copy_file_safe(src: Path, dst: Path) -> bool:
    """
    Safely copy a file with error handling.

    Args:
        src: Source file path
        dst: Destination file path

    Returns:
        bool: True if copy successful
    """
    try:
        src = Path(src)
        dst = Path(dst)

        # Ensure source file exists
        ensure_directory(dst.parent)

        # Copy the file
        shutil.copy2(src, dst)
        logger.debug(f"Copied {src} to {dst}")
        return True

    except Exception as exc:
        logger.error(f"Failed to copy {src} to {dst}: {exc}")
        return False


def find_files_by_pattern(directory: Path, pattern: str) -> List[Path]:
    """
    Find files matching a pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match

    Returns:
        List[Path]: List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    try:
        return list(directory.glob(pattern))
    except Exception as e:
        logger.error(f"Error searching for pattern {pattern} in {directory}: {e}")
        return []


def get_file_info(file_path: Path) -> Optional[dict]:
    """
    Get detailed file information.

    Args:
        file_path: Path to file

    Returns:
        dict: File information or None if error
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        stat = file_path.stat()

        return {
            "name": file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir(),
            "extension": file_path.suffix.lower(),
        }

    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return None
