"""
Image processing utilities
"""

import base64
from io import BytesIO
from PIL import Image
import requests
from typing import Union
import logging

logger = logging.getLogger(__name__)


def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert base64 string to PIL Image
    """
    try:
        # Handle data URL format
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]

        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        raise ValueError(f"Invalid base64 image data: {e}")


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert PIL Image to base64 string
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def download_image(url: str) -> Image.Image:
    """
    Download image from URL
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image.convert("RGB")
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
        raise ValueError(f"Failed to download image: {e}")


def validate_image_size(
    image: Image.Image, max_size: tuple = (2048, 2048)
) -> Image.Image:
    """
    Validate and resize image if too large
    """
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logger.info(f"Image resized to {image.size}")

    return image
