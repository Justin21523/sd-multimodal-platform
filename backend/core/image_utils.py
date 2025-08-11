# backend/core/image_utils.py
"""
Image Processing Utilities

Provides common image operations for the SD platform:
- Image saving and loading with metadata
- Format conversions and optimizations
- Thumbnail generation
- Batch processing utilities
"""

import io
import uuid
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import base64
import logging

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing operations for the SD platform"""

    def __init__(self):
        self.output_dir = Path(Settings.output_dir)
        self.temp_dir = Path(Settings.temp_dir)

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def save_image_with_metadata(
        self,
        image: Image.Image,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "PNG",
    ) -> Tuple[str, str]:
        """
        Save image with generation metadata

        Args:
            image: PIL Image to save
            filename: Custom filename (auto-generated if None)
            metadata: Generation parameters to embed
            format: Image format (PNG, JPEG, WEBP)

        Returns:
            Tuple of (filename, full_path)
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"sd_{timestamp}_{unique_id}.{format.lower()}"

        full_path = self.output_dir / filename

        try:
            # Prepare metadata for PNG
            if format.upper() == "PNG" and metadata:
                pnginfo = PngInfo()
                for key, value in metadata.items():
                    pnginfo.add_text(str(key), str(value))
                image.save(full_path, format=format, pnginfo=pnginfo, optimize=True)
            else:
                image.save(full_path, format=format, optimize=True)

            logger.info(f"✅ Image saved: {filename}")
            return filename, str(full_path)

        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise e

    def save_batch_images(
        self,
        images: List[Image.Image],
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "PNG",
    ) -> List[Tuple[str, str]]:
        """
        Save multiple images with batch naming

        Args:
            images: List of PIL Images
            metadata: Shared metadata for all images
            format: Image format

        Returns:
            List of (filename, full_path) tuples
        """
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = str(uuid.uuid4())[:8]

        for i, image in enumerate(images):
            filename = f"sd_batch_{timestamp}_{batch_id}_{i+1:02d}.{format.lower()}"

            # Add batch info to metadata
            batch_metadata = metadata.copy() if metadata else {}
            batch_metadata.update(
                {"batch_id": batch_id, "batch_index": i + 1, "batch_total": len(images)}
            )

            filename, full_path = self.save_image_with_metadata(
                image, filename, batch_metadata, format
            )
            results.append((filename, full_path))

        return results

    def create_thumbnail(
        self,
        image: Image.Image,
        size: Tuple[int, int] = (256, 256),
        maintain_aspect: bool = True,
    ) -> Image.Image:
        """
        Create thumbnail preserving aspect ratio

        Args:
            image: Source image
            size: Target thumbnail size
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Thumbnail image
        """
        if maintain_aspect:
            # Use PIL's thumbnail method which maintains aspect ratio
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            return thumbnail
        else:
            # Force exact size
            return image.resize(size, Image.Resampling.LANCZOS)

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Convert PIL Image to base64 string

        Args:
            image: PIL Image
            format: Image format for encoding

        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"

    def base64_to_image(self, base64_str: str) -> Image.Image:
        """
        Convert base64 string to PIL Image

        Args:
            base64_str: Base64 encoded image string

        Returns:
            PIL Image
        """
        # Remove data URL prefix if present
        if base64_str.startswith("data:"):
            base64_str = base64_str.split(",")[1]

        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))

    def calculate_image_hash(self, image: Image.Image) -> str:
        """
        Calculate perceptual hash for image deduplication

        Args:
            image: PIL Image

        Returns:
            MD5 hash string
        """
        # Convert to bytes for hashing
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return hashlib.md5(img_bytes).hexdigest()

    def optimize_image_size(
        self, image: Image.Image, max_size: int = 2048, quality: int = 95
    ) -> Image.Image:
        """
        Optimize image size while maintaining quality

        Args:
            image: Source image
            max_size: Maximum dimension (width or height)
            quality: JPEG quality (for JPEG format)

        Returns:
            Optimized image
        """
        width, height = image.size

        # Check if resizing is needed
        if max(width, height) > max_size:
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)

            # Resize with high-quality resampling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(
                f"Image resized from {width}x{height} to {new_width}x{new_height}"
            )

        return image

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Clean up temporary files older than specified age

        Args:
            max_age_hours: Maximum age in hours before deletion
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0

        try:
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} temporary files")

        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
