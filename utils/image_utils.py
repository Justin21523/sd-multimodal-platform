# utils/image_utils.py
"""
SD Multi-Modal Platform - Image Processing Utilities
Extended image processing utilities for ControlNet, img2img, and asset management
"""
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import io
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pathlib import Path
import cv2
import base64
import numpy as np

logger = logging.getLogger(__name__)


# 解決 LANCZOS 屬性問題
try:
    LANCZOS_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_FILTER = Image.ANTIALIAS  # type: ignore


class ImageProcessor:
    """圖像處理工具類"""

    def __init__(self):
        self.supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    def pil_to_numpy(self, image: Image.Image) -> np.ndarray:
        """PIL Image 轉 numpy array"""
        return np.array(image)

    def numpy_to_pil(self, array: np.ndarray) -> Image.Image:
        """numpy array 轉 PIL Image"""
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)

    def resize_image(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """調整圖像大小"""
        return image.resize(size, LANCZOS_FILTER)

    def crop_center(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """中心裁剪"""
        width, height = image.size
        new_width, new_height = size

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        return image.crop((left, top, right, bottom))

    def enhance_image(
        self,
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
    ) -> Image.Image:
        """圖像增強"""
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)

        return image

    def to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """圖像轉 base64"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()

    def from_base64(self, base64_str: str) -> Image.Image:
        """base64 轉圖像"""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))

    def ensure_rgb(self, image: Image.Image) -> Image.Image:
        """確保圖像為 RGB 格式"""
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def normalize_size(
        self, width: int, height: int, multiple: int = 8
    ) -> Tuple[int, int]:
        """標準化尺寸為指定倍數"""
        width = (width // multiple) * multiple
        height = (height // multiple) * multiple
        return width, height


def base64_to_pil_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image with validation"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        return image

    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def pil_image_to_base64(
    image: Image.Image, format: str = "PNG", quality: int = 95
) -> str:
    """Convert PIL Image to base64 string"""
    try:
        buffer = BytesIO()
        save_kwargs = {"format": format}

        if format.upper() == "JPEG":
            save_kwargs["quality"] = quality  # type: ignore
            save_kwargs["optimize"] = True  # type: ignore
            # Convert RGBA to RGB for JPEG
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background

        image.save(buffer, **save_kwargs)
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode()

    except Exception as e:
        raise ValueError(f"Failed to encode image to base64: {str(e)}")


def prepare_img2img_image(
    image: Image.Image,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Image.Image:
    """
    Prepare image for img2img generation with SD-compatible dimensions
    """
    try:
        # Use original dimensions if not specified
        if target_width is None:
            target_width = image.width
        if target_height is None:
            target_height = image.height

        # Ensure SD-compatible dimensions (multiple of 8)
        target_width = ((target_width + 7) // 8) * 8
        target_height = ((target_height + 7) // 8) * 8

        # Resize with high-quality resampling
        if image.size != (target_width, target_height):
            image = image.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )

        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    except Exception as e:
        logger.error(f"Failed to prepare img2img image: {str(e)}")
        raise


def prepare_inpaint_mask(
    init_image: Image.Image,
    mask_image: Image.Image,
    blur_radius: int = 4,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Tuple[Image.Image, Image.Image]:
    """
    Prepare images for inpainting with proper mask processing
    """
    try:
        # Determine target dimensions
        if target_width is None:
            target_width = init_image.width
        if target_height is None:
            target_height = init_image.height

        # Ensure SD-compatible dimensions
        target_width = ((target_width + 7) // 8) * 8
        target_height = ((target_height + 7) // 8) * 8

        # Prepare init image
        processed_init = prepare_img2img_image(init_image, target_width, target_height)

        # Process mask
        processed_mask = mask_image.convert("L")  # Convert to grayscale

        # Resize mask to match init image
        if processed_mask.size != (target_width, target_height):
            processed_mask = processed_mask.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )

        # Apply blur to mask edges for smoother blending
        if blur_radius > 0:
            processed_mask = processed_mask.filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )

        # Ensure mask is binary (0 or 255)
        mask_array = np.array(processed_mask)
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        processed_mask = Image.fromarray(mask_array, mode="L")

        return processed_init, processed_mask

    except Exception as e:
        logger.error(f"Failed to prepare inpaint mask: {str(e)}")
        raise


def preprocess_controlnet_image(
    image: Image.Image,
    target_width: int = 512,
    target_height: int = 512,
    maintain_aspect: bool = True,
) -> Image.Image:
    """
    Preprocess image for ControlNet conditioning
    """
    try:
        # Ensure SD-compatible dimensions
        target_width = ((target_width + 7) // 8) * 8
        target_height = ((target_height + 7) // 8) * 8

        if maintain_aspect:
            # Calculate aspect ratio preserving resize
            aspect_ratio = image.width / image.height
            target_aspect = target_width / target_height

            if aspect_ratio > target_aspect:
                # Image is wider, fit to width
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # Image is taller, fit to height
                new_height = target_height
                new_width = int(target_height * aspect_ratio)

            # Ensure dimensions are multiples of 8
            new_width = ((new_width + 7) // 8) * 8
            new_height = ((new_height + 7) // 8) * 8

            # Resize image
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Pad to target size with black
            padded = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            padded.paste(resized, (paste_x, paste_y))

            return padded
        else:
            # Direct resize without maintaining aspect ratio
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    except Exception as e:
        logger.error(f"Failed to preprocess ControlNet image: {str(e)}")
        raise


def create_canny_edges(
    image: Image.Image,
    low_threshold: int = 50,
    high_threshold: int = 200,
    blur_radius: int = 0,
) -> Image.Image:
    """
    Create Canny edge detection for ControlNet
    """
    try:
        # Convert to numpy array
        image_array = np.array(image.convert("RGB"))

        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur if specified
        if blur_radius > 0:
            gray = cv2.GaussianBlur(gray, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Convert back to PIL Image
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

    except Exception as e:
        logger.error(f"Failed to create Canny edges: {str(e)}")
        raise


def create_depth_map(image: Image.Image) -> Image.Image:
    """
    Create depth map from image (simplified version)
    Note: In production, use MiDaS or similar depth estimation models
    """
    try:
        # Convert to grayscale and invert for pseudo-depth
        gray = image.convert("L")
        depth_array = np.array(gray)

        # Simple depth estimation: darker = closer, lighter = farther
        depth_array = 255 - depth_array

        # Apply Gaussian blur for smoother depth transitions
        depth_array = cv2.GaussianBlur(depth_array, (15, 15), 0)

        # Convert back to RGB
        depth_rgb = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_rgb)

    except Exception as e:
        logger.error(f"Failed to create depth map: {str(e)}")
        raise


def optimize_asset_image(
    image: Image.Image, max_size: Tuple[int, int] = (2048, 2048), quality: int = 95
) -> Image.Image:
    """
    Optimize image for asset storage
    """
    try:
        # Resize if too large
        if image.width > max_size[0] or image.height > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Optimize based on content
        if image.mode == "RGBA":
            # Check if alpha channel is actually used
            alpha = image.split()[-1]
            if alpha.getextrema() == (255, 255):  # No transparency
                image = image.convert("RGB")

        return image

    except Exception as e:
        logger.error(f"Failed to optimize asset image: {str(e)}")
        raise


def create_thumbnail(
    image: Image.Image, size: Tuple[int, int] = (256, 256), maintain_aspect: bool = True
) -> Image.Image:
    """
    Create thumbnail with optional aspect ratio preservation
    """
    try:
        if maintain_aspect:
            # Create thumbnail maintaining aspect ratio
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)

            # Center on background if needed
            if thumbnail.size != size:
                background = Image.new("RGB", size, (255, 255, 255))
                paste_x = (size[0] - thumbnail.width) // 2
                paste_y = (size[1] - thumbnail.height) // 2
                background.paste(thumbnail, (paste_x, paste_y))
                thumbnail = background

            return thumbnail
        else:
            # Direct resize to exact size
            return image.resize(size, Image.Resampling.LANCZOS)

    except Exception as e:
        logger.error(f"Failed to create thumbnail: {str(e)}")
        raise


def validate_image_format(image: Image.Image) -> bool:
    """
    Validate image format and properties for SD processing
    """
    try:
        # Check dimensions
        if image.width < 64 or image.height < 64:
            return False

        if image.width > 4096 or image.height > 4096:
            return False

        # Check mode
        if image.mode not in ("RGB", "RGBA", "L"):
            return False

        # Check if image is corrupted
        image.verify()

        return True

    except Exception:
        return False


def create_test_image(
    width: int, height: int, mode: str = "RGB", fill_color=None
) -> Image.Image:
    """Create a test image for testing purposes"""
    if mode == "RGB":
        color = fill_color or (128, 128, 128)  # Gray
    elif mode == "L":
        color = fill_color or 128  # Gray
    elif mode == "RGBA":
        color = fill_color or (128, 128, 128, 255)  # Gray with alpha
    else:
        color = fill_color or 128

    return Image.new(mode, (width, height), color)


def get_image_info(image: Union[Image.Image, str]) -> dict:
    """
    Get comprehensive image information
    """
    try:
        if isinstance(image, str):
            image = Image.open(image)

        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_ratio": f"{image.width}:{image.height}",
            "megapixels": round(image.width * image.height / 1000000, 2),
            "aspect_ratio": round(image.width / image.height, 3),
            "is_square": image.width == image.height,
            "is_landscape": image.width > image.height,
            "is_portrait": image.height > image.width,
            "sd_compatible": image.width % 8 == 0 and image.height % 8 == 0,
        }

    except Exception as e:
        logger.error(f"Failed to get image info: {str(e)}")
        return {}


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
