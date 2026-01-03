# app/services/postprocess/upscale_service.py
import logging
import asyncio
import time
import uuid
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import torch
from PIL import Image


from app.config import settings
from utils.logging_utils import get_logger
from utils.image_utils import ImageProcessor
from utils.file_utils import FileManager
from utils.metadata_utils import ImageMetadata, MetadataManager

try:
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from basicsr.archs.rrdbnet_arch import RRDBNet

    REALESRGAN_AVAILABLE = True
except Exception:  # pragma: no cover
    RealESRGANer = None  # type: ignore
    SRVGGNetCompact = None  # type: ignore
    RRDBNet = None  # type: ignore
    REALESRGAN_AVAILABLE = False


class RealESRGANWrapper:
    """Real-ESRGAN 包裝器，解決 API 不一致問題"""

    def __init__(self, model_name: str, model_path: str, scale: int = 4):
        self.model_name = model_name
        self.model_path = model_path
        self.scale = scale
        self.upsampler = None
        self._tile_size = 512  # 內部維護 tile 大小

    def initialize(self):
        """初始化 Real-ESRGAN 模型"""
        if not REALESRGAN_AVAILABLE:
            raise ImportError("Real-ESRGAN not available")

        # 根據模型選擇適當的網路架構
        if "anime" in self.model_name.lower():
            # 動漫模型使用較少的 block
            model = RRDBNet(  # type: ignore
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,  # 動漫模型
                num_grow_ch=32,
                scale=self.scale,
            )
        else:
            # 通用模型
            model = RRDBNet(  # type: ignore
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,  # 通用模型
                num_grow_ch=32,
                scale=self.scale,
            )

        # 創建 upsampler，不使用 tile 參數（在 enhance 方法中處理）
        self.upsampler = RealESRGANer(  # type: ignore
            scale=self.scale,
            model_path=self.model_path,
            model=model,
            tile=0,  # 設為 0 表示不分塊
            tile_pad=10,
            pre_pad=0,
            half=True,  # 使用半精度
            gpu_id=0,
        )

    @property
    def tile(self):
        """獲取當前 tile 大小"""
        return self._tile_size

    @tile.setter
    def tile(self, value):
        """設置 tile 大小"""
        self._tile_size = value
        if self.upsampler:
            # 重新初始化 upsampler 以應用新的 tile 設置
            self.upsampler.tile = value  # type: ignore

    def enhance(
        self,
        img,
        outscale=None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        abort_check: Optional[Callable[[], None]] = None,
    ):
        """
        增強圖像解析度

        Args:
            img: 輸入圖像 (numpy array, BGR 格式)
            outscale: 輸出縮放倍數

        Returns:
            (enhanced_img, _)
        """
        if outscale is None:
            outscale = self.scale

        if abort_check is not None:
            abort_check()

        # 如果需要分塊處理大圖像
        if self._tile_size > 0:
            # 手動實現分塊處理
            enhanced_img = self._tile_enhance(
                img,
                outscale,
                progress_callback=progress_callback,
                abort_check=abort_check,
            )
        else:
            # 直接處理整張圖像
            if abort_check is not None:
                abort_check()
            enhanced_img, _ = self.upsampler.enhance(img, outscale=outscale)  # type: ignore
            if progress_callback is not None:
                try:
                    progress_callback(0, 1)
                except Exception:
                    pass

        return enhanced_img, None

    def _tile_enhance(
        self,
        img,
        outscale,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        abort_check: Optional[Callable[[], None]] = None,
    ):
        """分塊處理大圖像"""
        h, w, c = img.shape
        tile_size = self._tile_size
        overlap = 32  # 重疊像素，避免接縫

        if abort_check is not None:
            abort_check()

        # 如果圖像小於 tile 大小，直接處理
        if h <= tile_size and w <= tile_size:
            if abort_check is not None:
                abort_check()
            enhanced_img, _ = self.upsampler.enhance(img, outscale=outscale)  # type: ignore
            if progress_callback is not None:
                try:
                    progress_callback(0, 1)
                except Exception:
                    pass
            return enhanced_img

        # 計算需要的 tile 數量
        tiles_h = (h - 1) // (tile_size - overlap) + 1
        tiles_w = (w - 1) // (tile_size - overlap) + 1
        total_tiles = max(1, int(tiles_h * tiles_w))

        # 創建輸出圖像
        enhanced_h = h * outscale
        enhanced_w = w * outscale
        enhanced_img = np.zeros((enhanced_h, enhanced_w, c), dtype=np.uint8)

        tile_index = 0
        for i in range(tiles_h):
            for j in range(tiles_w):
                if abort_check is not None:
                    abort_check()
                # 計算 tile 邊界
                start_h = i * (tile_size - overlap)
                end_h = min(start_h + tile_size, h)
                start_w = j * (tile_size - overlap)
                end_w = min(start_w + tile_size, w)

                # 提取 tile
                tile = img[start_h:end_h, start_w:end_w]

                # 處理 tile
                enhanced_tile, _ = self.upsampler.enhance(tile, outscale=outscale)  # type: ignore

                # 計算輸出位置
                out_start_h = start_h * outscale
                out_end_h = end_h * outscale
                out_start_w = start_w * outscale
                out_end_w = end_w * outscale

                # 放置處理後的 tile
                enhanced_img[out_start_h:out_end_h, out_start_w:out_end_w] = (
                    enhanced_tile
                )

                if progress_callback is not None:
                    try:
                        progress_callback(tile_index, total_tiles)
                    except Exception:
                        pass
                tile_index += 1

        return enhanced_img


logger = logging.getLogger(__name__)


class UpscaleService:
    """
    Image upscaling service using Real-ESRGAN

    Supports:
    - Multiple Real-ESRGAN models (x2, x4, x8)
    - Batch processing
    - Memory optimization
    - Custom tile sizes for large images
    - CUDA acceleration
    """

    def __init__(self):
        self.is_initialized = False
        self.current_model = None
        self.upsampler: Optional[RealESRGANer] = None  # type: ignore
        self.device = settings.DEVICE

        weights_dir = Path(settings.UPSCALE_MODELS_PATH)

        # Available models
        self.available_models = {
            "RealESRGAN_x4plus": {
                "model_name": "RealESRGAN_x4plus",
                "model_path": str(weights_dir / "RealESRGAN_x4plus.pth"),
                "scale": 4,
                "netscale": 4,
                "tile": 512,
                "tile_pad": 10,
                "pre_pad": 0,
                "half": True if self.device == "cuda" else False,
            },
            "RealESRGAN_x2plus": {
                "model_name": "RealESRGAN_x2plus",
                "model_path": str(weights_dir / "RealESRGAN_x2plus.pth"),
                "scale": 2,
                "netscale": 2,
                "tile": 400,
                "tile_pad": 10,
                "pre_pad": 0,
                "half": True if self.device == "cuda" else False,
            },
            "RealESRGAN_x4plus_anime_6B": {
                "model_name": "RealESRGAN_x4plus_anime_6B",
                "model_path": str(weights_dir / "RealESRGAN_x4plus_anime_6B.pth"),
                "scale": 4,
                "netscale": 4,
                "tile": 512,
                "tile_pad": 10,
                "pre_pad": 0,
                "half": True if self.device == "cuda" else False,
            },
            "RealESRNet_x4plus": {
                "model_name": "RealESRNet_x4plus",
                "model_path": str(weights_dir / "RealESRNet_x4plus.pth"),
                "scale": 4,
                "netscale": 4,
                "tile": 512,
                "tile_pad": 10,
                "pre_pad": 0,
                "half": True if self.device == "cuda" else False,
            },
        }

        # Service utilities
        self.image_processor = ImageProcessor()
        self.file_manager = FileManager()
        self.metadata_manager = MetadataManager()

        # Performance tracking
        self.total_upscales = 0
        self.total_time = 0.0
        self.startup_time = None

    async def initialize(self, model_name: str = "RealESRGAN_x4plus"):
        """Initialize the upscaling service with specified model"""
        try:
            start_time = time.time()
            logger.info("🔍 Initializing Upscale service...")

            if not REALESRGAN_AVAILABLE:
                logger.warning(
                    "Real-ESRGAN not available, using fallback implementation"
                )
                if model_name in self.available_models:
                    self.current_model = model_name
                else:
                    self.current_model = "RealESRGAN_x4plus"
                self.is_initialized = True
                return

            # Load model
            await self._load_model(model_name)

            # Test with small image
            if not settings.MOCK_GENERATION:
                await self._warmup_model()

            self.startup_time = time.time() - start_time
            self.is_initialized = True

            logger.info(f"✅ Upscale service initialized in {self.startup_time:.2f}s")
            logger.info(f"📋 Model: {self.current_model}, Device: {self.device}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Upscale service: {e}")
            raise

    def _resolve_weights_path(self, expected_path: Path) -> Path:
        """Resolve model weights path, supporting legacy installer layouts."""
        if expected_path.exists():
            return expected_path

        weights_dir = Path(settings.UPSCALE_MODELS_PATH)
        legacy_candidates = [
            weights_dir
            / "real-esrgan"
            / "experiments"
            / "pretrained_models"
            / expected_path.name,
            weights_dir / "real-esrgan" / expected_path.name,
        ]
        for candidate in legacy_candidates:
            if candidate.exists():
                logger.info(f"Using legacy upscale weight path: {candidate}")
                return candidate

        raise FileNotFoundError(
            f"Upscale model weights not found: {expected_path} (expected under {weights_dir}). "
            "Run `python scripts/install_models.py --postprocess real-esrgan-x4` (and/or real-esrgan-x2, real-esrgan-anime) to install."
        )

    async def _load_model(self, model_name: str):
        """Load specified Real-ESRGAN model"""
        try:
            config = self.available_models[model_name]
            model_path = self._resolve_weights_path(Path(config["model_path"]))

            # 使用包裝器而非直接使用 RealESRGANer
            self.upsampler = RealESRGANWrapper(
                model_name=model_name, model_path=str(model_path), scale=config["scale"]
            )

            self.upsampler.initialize()  # type: ignore
            self.current_model = model_name

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    async def _download_model(self, model_name: str, model_path: Path):
        """Download Real-ESRGAN model weights"""
        try:
            logger.info(f"📥 Downloading {model_name} model...")

            # Create weights directory
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Model download URLs
            download_urls = {
                "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
            }

            if model_name not in download_urls:
                raise ValueError(f"No download URL for model {model_name}")

            url = download_urls[model_name]

            # Download using aiohttp or requests
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(model_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        logger.info(f"✅ Model {model_name} downloaded successfully")
                    else:
                        raise Exception(
                            f"Failed to download model: HTTP {response.status}"
                        )

        except Exception as e:
            logger.error(f"❌ Failed to download model {model_name}: {e}")
            raise

    async def _warmup_model(self):
        """Warm up the model with a test upscale"""
        try:
            if not self.upsampler:
                return

            logger.info("🔥 Warming up upscale model...")

            # Create small test image
            test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

            # Run test upscale
            _, _ = self.upsampler.enhance(test_img, outscale=2)

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ Upscale model warmed up")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    async def upscale_image(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        scale: Optional[int] = None,
        model_name: Optional[str] = None,
        tile_size: Optional[int] = None,
        user_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        abort_check: Optional[Callable[[], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Upscale image using Real-ESRGAN

        Args:
            image: Input image (PIL Image, file path, or numpy array)
            scale: Upscaling factor (2, 4, or 8)
            model_name: Model to use for upscaling
            tile_size: Tile size for processing large images
            user_id: User identifier for organization
            **kwargs: Additional parameters

        Returns:
            Dictionary containing upscaling results and metadata
        """
        try:
            start_time = time.time()

            if not self.is_initialized:
                await self.initialize()

            # Process input image
            input_image, img_array = await self._process_input_image(image)

            # Determine model and scale
            effective_model = model_name or self.current_model or "RealESRGAN_x4plus"
            if effective_model not in self.available_models:
                effective_model = "RealESRGAN_x4plus"

            if REALESRGAN_AVAILABLE:
                if effective_model != self.current_model:
                    await self._load_model(effective_model)
            else:
                # Fallback mode: don't try to load weights, but keep metadata stable.
                self.current_model = effective_model

            config = self.available_models.get(self.current_model or effective_model) or self.available_models["RealESRGAN_x4plus"]

            if scale is None:
                scale = config["scale"]  # type: ignore[index]

            # Validate scale
            max_scale = config["scale"]  # type: ignore[index]
            if scale > max_scale:
                logger.warning(
                    f"Requested scale {scale} > model max {max_scale}, using {max_scale}"
                )
                scale = max_scale

            logger.info(
                f"🔍 Starting upscaling: {input_image.width}x{input_image.height} -> scale x{scale}"
            )

            # Perform upscaling
            if settings.MOCK_GENERATION or not REALESRGAN_AVAILABLE:
                # Mock upscaling for testing
                if abort_check is not None:
                    abort_check()
                upscaled_array = await self._mock_upscale(img_array, scale)  # type: ignore
                await asyncio.sleep(0.1)  # Simulate processing time
            else:
                upscaled_array = await self._real_upscale(
                    img_array,
                    scale,
                    tile_size,
                    progress_callback=progress_callback,
                    abort_check=abort_check,
                )  # type: ignore

            # Convert back to PIL Image
            upscaled_image = Image.fromarray(upscaled_array)

            processing_time = time.time() - start_time

            # Process and save results
            results = await self._process_upscale_results(
                original_image=input_image,
                upscaled_image=upscaled_image,
                scale=scale,  # type: ignore
                model_name=str(self.current_model or effective_model),
                processing_time=processing_time,
                user_id=user_id,
            )

            # Update statistics
            self.total_upscales += 1
            self.total_time += processing_time

            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"✅ Upscaling completed in {processing_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"❌ Upscaling failed: {e}")

            # GPU cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise

    async def _process_input_image(
        self, image: Union[Image.Image, str, Path, np.ndarray]
    ) -> Tuple[Image.Image, np.ndarray]:
        """Process and validate input image"""
        try:
            # Handle different input types
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                pil_image = Image.open(image_path)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Ensure RGB format
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Convert to numpy array (BGR for OpenCV)
            img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Validate image size
            max_dimension = 4096
            if pil_image.width > max_dimension or pil_image.height > max_dimension:
                ratio = max_dimension / max(pil_image.width, pil_image.height)
                new_width = int(pil_image.width * ratio)
                new_height = int(pil_image.height * ratio)
                LANCZOS_FILTER = Image.Resampling.LANCZOS
                pil_image = pil_image.resize((new_width, new_height), LANCZOS_FILTER)
                img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                logger.warning(
                    f"Image resized to {new_width}x{new_height} (max size exceeded)"
                )

            return pil_image, img_array

        except Exception as e:
            logger.error(f"Failed to process input image: {e}")
            raise

    async def _real_upscale(
        self,
        img_array: np.ndarray,
        scale: int,
        tile_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        abort_check: Optional[Callable[[], None]] = None,
    ) -> np.ndarray:
        """Perform real upscaling using Real-ESRGAN"""
        try:
            if not self.upsampler:
                raise RuntimeError("Upsampler not initialized")

            # 設置 tile 大小
            if tile_size:
                original_tile = self.upsampler.tile
                self.upsampler.tile = tile_size

            # 執行放大
            upscaled_img, _ = self.upsampler.enhance(
                img_array,
                outscale=scale,
                progress_callback=progress_callback,
                abort_check=abort_check,
            )

            # 恢復原始 tile 大小
            if tile_size:
                self.upsampler.tile = original_tile

            # 轉換 BGR 到 RGB
            upscaled_rgb = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB)

            return upscaled_rgb

        except Exception as e:
            logger.error(f"Real upscaling failed: {e}")
            raise

    async def _mock_upscale(self, img_array: np.ndarray, scale: int) -> np.ndarray:
        """Mock upscaling for testing (simple bilinear interpolation)"""
        try:
            height, width = img_array.shape[:2]
            new_height, new_width = height * scale, width * scale

            # Convert BGR to RGB for PIL
            rgb_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_array)

            # Simple resize
            LANCZOS_FILTER = Image.Resampling.LANCZOS
            upscaled_pil = pil_image.resize((new_width, new_height), LANCZOS_FILTER)
            # Convert back to array
            upscaled_array = np.array(upscaled_pil)

            return upscaled_array

        except Exception as e:
            logger.error(f"Mock upscaling failed: {e}")
            raise

    async def _process_upscale_results(
        self,
        original_image: Image.Image,
        upscaled_image: Image.Image,
        scale: int,
        model_name: str,
        processing_time: float,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Process and save upscaling results"""
        try:
            task_id = f"upscale_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"upscale_{timestamp}_{scale}x_{model_name}.png"

            # Save upscaled image
            image_path = await self.file_manager.save_image(
                image=upscaled_image,
                filename=filename,
                subfolder="upscale",
                user_id=user_id,
            )

            # Create metadata
            metadata = ImageMetadata(
                filename=filename,
                prompt=f"Upscaled {scale}x using {model_name}",
                model=model_name,
                width=upscaled_image.width,
                height=upscaled_image.height,
                generation_time=processing_time,
                task_id=task_id,
                task_type="upscale",
                user_id=user_id,
                additional_params={
                    "original_size": f"{original_image.width}x{original_image.height}",
                    "upscaled_size": f"{upscaled_image.width}x{upscaled_image.height}",
                    "scale_factor": scale,
                    "model_used": model_name,
                    "processing_time": processing_time,
                },
            )

            # Save metadata
            metadata_path = await self.metadata_manager.save_metadata(
                metadata=metadata, task_id=task_id
            )

            # Calculate VRAM usage
            vram_used = None
            if torch.cuda.is_available():
                vram_used = f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"

            return {
                "success": True,
                "task_id": task_id,
                "parameters": {
                    "model": model_name,
                    "scale": scale,
                    "original_size": f"{original_image.width}x{original_image.height}",
                    "upscaled_size": f"{upscaled_image.width}x{upscaled_image.height}",
                },
                "result": {
                    "image_path": str(image_path),
                    "image_url": f"/outputs/upscale/{filename}",
                    "metadata_path": str(metadata_path),
                    "original_width": original_image.width,
                    "original_height": original_image.height,
                    "upscaled_width": upscaled_image.width,
                    "upscaled_height": upscaled_image.height,
                    "scale_factor": scale,
                    "processing_time": round(processing_time, 2),
                    "model_used": model_name,
                    "vram_used": vram_used,
                    "device": self.device,
                },
                "metadata": {
                    "task_type": "upscale",
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                },
            }

        except Exception as e:
            logger.error(f"Failed to process upscale results: {e}")
            raise

    async def batch_upscale(
        self,
        images: List[Union[Image.Image, str, Path]],
        scale: int = 4,
        model_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch upscale multiple images"""
        try:
            start_time = time.time()
            logger.info(f"🔍 Starting batch upscale: {len(images)} images")

            results = []
            failed_images = []

            for i, image in enumerate(images):
                try:
                    logger.info(f"Processing image {i+1}/{len(images)}")

                    result = await self.upscale_image(
                        image=image, scale=scale, model_name=model_name, user_id=user_id
                    )

                    results.append(result)

                except Exception as e:
                    logger.error(f"Failed to upscale image {i+1}: {e}")
                    failed_images.append({"index": i, "error": str(e)})

            total_time = time.time() - start_time

            return {
                "success": True,
                "batch_id": f"batch_upscale_{int(time.time()*1000)}",
                "total_images": len(images),
                "successful_images": len(results),
                "failed_images": len(failed_images),
                "total_time": round(total_time, 2),
                "results": results,
                "failures": failed_images,
            }

        except Exception as e:
            logger.error(f"Batch upscale failed: {e}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        avg_time = self.total_time / max(1, self.total_upscales)

        return {
            "service": "upscale",
            "initialized": self.is_initialized,
            "current_model": self.current_model,
            "device": self.device,
            "available_models": list(self.available_models.keys()),
            "realesrgan_available": REALESRGAN_AVAILABLE,
            "statistics": {
                "total_upscales": self.total_upscales,
                "total_time": round(self.total_time, 2),
                "average_time": round(avg_time, 2),
                "startup_time": round(self.startup_time or 0, 2),
            },
            "memory": {
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_allocated": (
                    f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                    if torch.cuda.is_available()
                    else None
                ),
                "gpu_memory_reserved": (
                    f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }

    async def cleanup(self):
        """Clean up resources and memory"""
        try:
            logger.info("🧹 Cleaning up Upscale service...")

            if self.upsampler is not None:
                del self.upsampler
                self.upsampler = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self.is_initialized = False
            self.current_model = None

            logger.info("✅ Upscale service cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def switch_model(self, model_name: str):
        """Switch to a different upscaling model"""
        try:
            if self.current_model == model_name:
                logger.info(f"Model {model_name} already loaded")
                return

            logger.info(f"🔄 Switching from {self.current_model} to {model_name}")

            # Cleanup current model
            if self.upsampler is not None:
                del self.upsampler
                self.upsampler = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load new model
            await self._load_model(model_name)

            logger.info(f"✅ Successfully switched to {model_name}")

        except Exception as e:
            logger.error(f"Failed to switch to model {model_name}: {e}")
            raise


# Global service instance
_upscale_service: Optional[UpscaleService] = None


async def get_upscale_service() -> UpscaleService:
    """Get global upscale service instance (singleton pattern)"""
    global _upscale_service
    if _upscale_service is None:
        _upscale_service = UpscaleService()
        await _upscale_service.initialize()
    return _upscale_service


async def cleanup_upscale_service():
    """Cleanup global upscale service"""
    global _upscale_service
    if _upscale_service is not None:
        await _upscale_service.cleanup()
        _upscale_service = None
