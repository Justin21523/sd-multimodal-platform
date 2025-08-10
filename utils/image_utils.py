# utils/image_utils.py
"""
SD Multi-Modal Platform - Image Processing Utilities
Phase 1: Image Optimization and Metadata Extraction
"""

import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageOps
import cv2
import base64
import numpy as np


def optimize_image(
    image: Image.Image,
    quality: int = 95,
    optimize: bool = True,
    max_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """Optimize an image for storage and processing."""

    # Transform to RGB if necessary
    if image.mode in ("RGBA", "P"):
        # Keep transparency if RGBA, otherwise convert to RGB
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(
            image, mask=image.split()[-1] if image.mode == "RGBA" else None
        )
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if max_size is provided
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Remove EXIF orientation
    result = ImageOps.exif_transpose(image)
    image = result or image  # Ensure we have a valid image

    return image


def get_image_info(image_path: Path) -> Dict[str, Any]:
    """Extract metadata and basic information from an image file."""
    try:
        with Image.open(image_path) as img:
            info = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "file_size": image_path.stat().st_size,
                "aspect_ratio": round(img.width / img.height, 2),
            }

            # Check if the image has EXIF data
            exif_method = getattr(img, "getexif", None)
            if callable(exif_method):
                has_exif = bool(exif_method())
            else:
                has_exif = False
            info["has_exif"] = has_exif

        return info

    except Exception as exc:
        return {"error": str(exc)}


def resize_image(
    image: Image.Image, target_width: int, target_height: int, method: str = "lanczos"
) -> Image.Image:
    """Resize an image to specified dimensions with a given resampling method."""

    # Ensure dimensions are multiples of 8 for compatibility
    target_width = (target_width // 8) * 8
    target_height = (target_height // 8) * 8

    # Define resampling methods
    resample_methods = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }

    resample = resample_methods.get(method, Image.Resampling.LANCZOS)

    return image.resize((target_width, target_height), resample)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert an image to a base64-encoded string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")
