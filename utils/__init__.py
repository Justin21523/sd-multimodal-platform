# utils/__init__.py
"""
SD Multi-Modal Platform - Utilities Package
Phase 1: Common Utilities for Image Processing, Logging, File Operations, and Metadata Management
"""

from .logging_utils import setup_logging, get_request_logger
from .image_utils import optimize_image, get_image_info, resize_image, image_to_base64
from .file_utils import ensure_directory, cleanup_old_files, safe_filename
from .metadata_utils import (
    save_metadata_json,
    load_metadata_json,
    find_metadata_by_seed,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_request_logger",
    # Image processing
    "optimize_image",
    "get_image_info",
    "resize_image",
    "image_to_base64",
    # File operations
    "ensure_directory",
    "cleanup_old_files",
    "safe_filename",
    # Metadata management
    "save_metadata_json",
    "load_metadata_json",
    "find_metadata_by_seed",
]
