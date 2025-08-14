# app/services/postprocess/face_restore_service.py
import asyncio
import time
import uuid
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import torch
from PIL import Image

try:
    # Try to import GFPGAN
    from gfpgan import GFPGANer

    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGANer = None
    GFPGAN_AVAILABLE = False


# Try to import CodeFormer
from codeformer import CodeFormer
CODEFORMER_AVAILABLE = True


from app.config import settings
from utils.logging_utils import get_logger
from utils.image_utils import ImageProcessor
from utils.file_utils import FileManager
from utils.metadata_utils import ImageMetadata, MetadataManager

logger = get_logger(__name__)

# 正確的 CodeFormer 導入和使用
try:
    import sys
    import os
    from basicsr.utils import imwrite

    # CodeFormer 的正確導入方式
    codeformer_path = "path/to/CodeFormer"  # CodeFormer 專案路徑
    if codeformer_path not in sys.path:
        sys.path.append(codeformer_path)

    from codeformer import CodeFormer as CodeFormerModel
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    CODEFORMER_AVAILABLE = True
except ImportError:
    CodeFormerModel = None
    FaceRestoreHelper = None
    CODEFORMER_AVAILABLE = False

class CodeFormerWrapper:
    """CodeFormer 包裝器，提供標準化介面"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        self.net = None
        self.face_helper = None

    def initialize(self):
        """初始化 CodeFormer 模型"""
        if not CODEFORMER_AVAILABLE:
            raise ImportError("CodeFormer not available")

        # 載入 CodeFormer 模型
        self.net = CodeFormerModel( # type: ignore
            dim_embd=512,  # type: ignore
            codebook_size=1024,  #  type: ignore
            n_head=8,  # type: ignore
            n_layers=9, # type: ignore
            connect_list=['32', '64', '128', '256'] # type: ignore
        ).to(self.device) # type: ignore

        # 載入預訓練權重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['params_ema'])
        self.net.eval()

        # 初始化人臉檢測輔助器
        self.face_helper = FaceRestoreHelper( # type: ignore
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device
        )

    def enhance(self,
                img_array: np.ndarray,
                w: float = 0.5,  # 保真度權重，不是 fidelity_weight
                has_aligned: bool = False,
                only_center_face: bool = False,
                paste_back: bool = True):
        """
        增強圖像中的人臉

        Args:
            img_array: 輸入圖像 (BGR 格式)
            w: 保真度權重 (0.0-1.0)，數值越高越保留原始特徵
            has_aligned: 是否已對齊
            only_center_face: 是否只處理中心人臉
            paste_back: 是否貼回原圖

        Returns:
            (cropped_faces, restored_faces, restored_img)
        """

        self.face_helper.clean_all() # type: ignore
        self.face_helper.read_image(img_array) # type: ignore

        # 檢測人臉
        self.face_helper.get_face_landmarks_5( # type: ignore
            only_center_face=only_center_face,
            resize=640,
            eye_dist_threshold=5
        )

        # 對齊和裁剪人臉
        self.face_helper.align_warp_face() # type: ignore

        # 復原每個檢測到的人臉
        for cropped_face in self.face_helper.cropped_faces: # type: ignore
            # 預處理
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=w, adain=True)[0] # type: ignore
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'Failed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1)) # type: ignore

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face) # type: ignore

        # 貼回原圖
        if paste_back:
            self.face_helper.get_inverse_affine(None) # type: ignore
            restored_img = self.face_helper.paste_faces_to_input_image() # type: ignore
        else:
            restored_img = img_array

        return (self.face_helper.cropped_faces,  # type: ignore
                self.face_helper.restored_faces, # type: ignore
                restored_img)

class FaceRestoreService:
    """
    Face restoration service using GFPGAN and CodeFormer

    Supports:
    - GFPGAN for face restoration
    - CodeFormer for robust face restoration
    - Batch processing
    - Face detection and enhancement
    - CUDA acceleration
    """

    def __init__(self):
        self.is_initialized = False
        self.current_model = None
        self.restorer = None
        self.device = settings.DEVICE

        # Available models
        self.available_models = {
            "GFPGAN_v1.4": {
                "model_name": "GFPGAN_v1.4",
                "model_path": "weights/GFPGANv1.4.pth",
                "arch": "clean",
                "channel_multiplier": 2,
                "upscale": 2,
                "bg_upsampler": None,
            },
            "GFPGAN_v1.3": {
                "model_name": "GFPGAN_v1.3",
                "model_path": "weights/GFPGANv1.3.pth",
                "arch": "clean",
                "channel_multiplier": 2,
                "upscale": 2,
                "bg_upsampler": None,
            },
            "CodeFormer": {
                "model_name": "CodeFormer",
                "model_path": "weights/codeformer.pth",
                "arch": "CodeFormer",
                "upscale": 2,
                "fidelity_weight": 0.5,
            },
            "RestoreFormer": {
                "model_name": "RestoreFormer",
                "model_path": "weights/RestoreFormer.pth",
                "arch": "RestoreFormer",
                "upscale": 2,
            },
        }

        # Service utilities
        self.image_processor = ImageProcessor()
        self.file_manager = FileManager()
        self.metadata_manager = MetadataManager()

        # Performance tracking
        self.total_restorations = 0
        self.total_faces_processed = 0
        self.total_time = 0.0
        self.startup_time = None

    async def initialize(self, model_name: str = "GFPGAN_v1.4"):
        """Initialize the face restoration service with specified model"""
        try:
            start_time = time.time()
            logger.info("👤 Initializing Face Restore service...")

            if not GFPGAN_AVAILABLE and not CODEFORMER_AVAILABLE:
                logger.warning(
                    "Neither GFPGAN nor CodeFormer available, using fallback"
                )
                self.is_initialized = True
                return

            # Load model
            await self._load_model(model_name)

            # Test with small image
            if not settings.MOCK_GENERATION:
                await self._warmup_model() # type: ignore

            self.startup_time = time.time() - start_time
            self.is_initialized = True

            logger.info(
                f"✅ Face Restore service initialized in {self.startup_time:.2f}s"
            )
            logger.info(f"📋 Model: {self.current_model}, Device: {self.device}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Face Restore service: {e}")
            raise

    async def _load_model(self, model_name: str):
        """Load specified face restoration model"""
        try:
            config = self.available_models[model_name]
            model_path = Path(config["model_path"])

            if "CodeFormer" in model_name and CODEFORMER_AVAILABLE:
                # 使用修正後的 CodeFormer 包裝器
                self.restorer = CodeFormerWrapper(
                    model_path=str(model_path),
                    device=self.device
                )
                self.restorer.initialize()

            elif "GFPGAN" in model_name and GFPGAN_AVAILABLE:
                # GFPGAN 的正確使用方式
                self.restorer = GFPGANer( # type: ignore
                    model_path=str(model_path),
                    upscale=config["upscale"],
                    arch=config["arch"],
                    channel_multiplier=config["channel_multiplier"],
                    bg_upsampler=None
                )

            self.current_model = model_name

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    async def _download_model(self, model_name: str, model_path: Path):
        """Download face restoration model weights"""
        try:
            logger.info(f"📥 Downloading {model_name} model...")

            # Create weights directory
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Model download URLs
            download_urls = {
                "GFPGAN_v1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                "GFPGAN_v1.3": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                "CodeFormer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                "RestoreFormer": "https://github.com/wzhouxiff/RestoreFormer/releases/download/v1.0/RestoreFormer.pth",
            }

            if model_name not in download_urls:
                raise ValueError(f"No download URL for model {model_name}")

            url = download_urls[model_name]

            # Download using aiohttp
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
        """Warm up the model with a test restoration"""
        try:
            if not self.restorer:
                return

            logger.info("🔥 Warming up face restore model...")

            # Create test face image (simple face-like pattern)
            test_img = np.ones((128, 128, 3), dtype=np.uint8) * 128
            # Add simple face-like features
            cv2.circle(test_img, (40, 50), 5, (255, 255, 255), -1)  # Left eye
            cv2.circle(test_img, (88, 50), 5, (255, 255, 255), -1)  # Right eye
            cv2.ellipse(
                test_img, (64, 90), (15, 8), 0, 0, 180, (255, 255, 255), 2
            )  # Mouth

            # Run test restoration
            if hasattr(self.restorer, "enhance"):
                _, _, _ = self.restorer.enhance( # type: ignore
                    test_img, has_aligned=False, only_center_face=False, paste_back=True
                )

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ Face restore model warmed up")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    async def restore_faces(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        model_name: Optional[str] = None,
        upscale: int = 2,
        only_center_face: bool = False,
        has_aligned: bool = False,
        paste_back: bool = True,
        weight: float = 0.5,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Restore faces in image using GFPGAN or CodeFormer

        Args:
            image: Input image (PIL Image, file path, or numpy array)
            model_name: Model to use for face restoration
            upscale: Upscaling factor for the face restoration
            only_center_face: Only restore the center face
            has_aligned: Whether the input is already aligned
            paste_back: Whether to paste faces back to the original image
            weight: Fidelity weight for CodeFormer (0.0-1.0)
            user_id: User identifier for organization
            **kwargs: Additional parameters

        Returns:
            Dictionary containing restoration results and metadata
        """
        try:
            start_time = time.time()

            if not self.is_initialized:
                await self.initialize()

            # Process input image
            input_image, img_array = await self._process_input_image(image)

            # Switch model if requested
            if model_name and model_name != self.current_model:
                await self._load_model(model_name)

            logger.info(
                f"👤 Starting face restoration: {input_image.width}x{input_image.height}"
            )

            # Perform face restoration
            if settings.MOCK_GENERATION or not (
                GFPGAN_AVAILABLE or CODEFORMER_AVAILABLE
            ):
                # Mock restoration for testing
                restored_img, cropped_faces, restored_faces, face_count = (
                    await self._mock_restore(img_array)
                )
                await asyncio.sleep(0.1)  # Simulate processing time
            else:
                restored_img, cropped_faces, restored_faces, face_count = (
                    await self._real_restore(
                        img_array,
                        upscale,
                        only_center_face,
                        has_aligned,
                        paste_back,
                        weight,
                    )
                )

            # Convert back to PIL Image
            restored_image = Image.fromarray(
                cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            )

            processing_time = time.time() - start_time

            # Process and save results
            results = await self._process_restoration_results(
                original_image=input_image,
                restored_image=restored_image,
                cropped_faces=cropped_faces,
                restored_faces=restored_faces,
                face_count=face_count,
                model_name=self.current_model,  # type: ignore
                upscale=upscale,
                processing_time=processing_time,
                user_id=user_id,
            )

            # Update statistics
            self.total_restorations += 1
            self.total_faces_processed += face_count
            self.total_time += processing_time

            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"✅ Face restoration completed in {processing_time:.2f}s ({face_count} faces)"
            )

            return results

        except Exception as e:
            logger.error(f"❌ Face restoration failed: {e}")

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
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume BGR format from OpenCV
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Ensure RGB format
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Convert to numpy array (BGR for OpenCV/GFPGAN)
            img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Validate image size
            max_dimension = 2048
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

    async def _real_restore(
        self,
        img_array: np.ndarray,
        upscale: int,
        only_center_face: bool,
        has_aligned: bool,
        paste_back: bool,
        weight: float,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], int]: # type: ignore
        """Perform real face restoration using GFPGAN or CodeFormer"""
        try:
            if not self.restorer:
                raise RuntimeError("Face restorer not initialized")

            if "GFPGAN" in self.current_model:  # type: ignore
                # Use GFPGAN
                cropped_faces, restored_faces, restored_img = self.restorer.enhance( # type: ignore
                    img_array,
                    has_aligned=has_aligned,
                    only_center_face=only_center_face,
                    paste_back=paste_back,
                    weight=weight, # type: ignore
                )

            elif "CodeFormer" in self.current_model:  # type:ignore
                # Use CodeFormer
                cropped_faces, restored_faces, restored_img = self.restorer.enhance( # type: ignore
                    img_array,
                    fidelity_weight=weight, # type: ignore
                    has_aligned=has_aligned,
                    only_center_face=only_center_face,
                    paste_back=paste_back,
                )

            else:
                raise RuntimeError(f"Unknown model type: {self.current_model}")

            face_count = len(cropped_faces) if cropped_faces is not None else 0

             eturn restored_img, cropped_faces or [], restored_faces or [], face_count  # type: ignore

        except Exception as e:
            logger.error(f"Real face restoration failed: {e}")
            raise

    async def _mock_restore(
        self, img_array: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], int]:
        """Mock face restoration for testing"""
        try:
            # Simple enhancement: increase brightness and contrast slightly
            enhanced = cv2.convertScaleAbs(img_array, alpha=1.1, beta=10)

            # Simulate face detection (create dummy face crops)
            height, width = img_array.shape[:2]

            # Create a dummy face crop (center region)
            face_size = min(width, height) // 3
            x = (width - face_size) // 2
            y = (height - face_size) // 2

            cropped_face = img_array[y : y + face_size, x : x + face_size]
            restored_face = cv2.convertScaleAbs(cropped_face, alpha=1.2, beta=15)

            return enhanced, [cropped_face], [restored_face], 1

        except Exception as e:
            logger.error(f"Mock face restoration failed: {e}")
            raise

    async def _process_restoration_results(
        self,
        original_image: Image.Image,
        restored_image: Image.Image,
        cropped_faces: List[np.ndarray],
        restored_faces: List[np.ndarray],
        face_count: int,
        model_name: str,
        upscale: int,
        processing_time: float,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Process and save face restoration results"""
        try:
            task_id = f"face_restore_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_restore_{timestamp}_{model_name}_{face_count}faces.png"

            # Save restored image
            image_path = await self.file_manager.save_image(
                image=restored_image,
                filename=filename,
                subfolder="face_restore",
                user_id=user_id,
            )

            # Save individual faces if any
            face_images = []
            for i, (cropped, restored) in enumerate(zip(cropped_faces, restored_faces)):
                try:
                    # Convert face arrays to PIL Images
                    cropped_pil = Image.fromarray(
                        cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    )
                    restored_pil = Image.fromarray(
                        cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
                    )

                    # Save cropped face
                    cropped_filename = (
                        f"face_restore_{timestamp}_face_{i:02d}_original.png"
                    )
                    cropped_path = await self.file_manager.save_image(
                        image=cropped_pil,
                        filename=cropped_filename,
                        subfolder="face_restore/faces",
                        user_id=user_id,
                    )

                    # Save restored face
                    restored_filename = (
                        f"face_restore_{timestamp}_face_{i:02d}_restored.png"
                    )
                    restored_path = await self.file_manager.save_image(
                        image=restored_pil,
                        filename=restored_filename,
                        subfolder="face_restore/faces",
                        user_id=user_id,
                    )

                    face_images.append(
                        {
                            "face_index": i,
                            "original_face_path": str(cropped_path),
                            "original_face_url": f"/outputs/face_restore/faces/{cropped_filename}",
                            "restored_face_path": str(restored_path),
                            "restored_face_url": f"/outputs/face_restore/faces/{restored_filename}",
                            "width": restored_pil.width,
                            "height": restored_pil.height,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to save face {i}: {e}")

            # Create metadata
            metadata = ImageMetadata(
                filename=filename,
                prompt=f"Face restoration using {model_name}",
                model=model_name,
                width=restored_image.width,
                height=restored_image.height,
                generation_time=processing_time,
                task_id=task_id,
                task_type="face_restore",
                user_id=user_id,
                additional_params={
                    "original_size": f"{original_image.width}x{original_image.height}",
                    "restored_size": f"{restored_image.width}x{restored_image.height}",
                    "faces_detected": face_count,
                    "upscale_factor": upscale,
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
                    "upscale": upscale,
                    "original_size": f"{original_image.width}x{original_image.height}",
                    "restored_size": f"{restored_image.width}x{restored_image.height}",
                    "faces_detected": face_count,
                },
                "result": {
                    "image_path": str(image_path),
                    "image_url": f"/outputs/face_restore/{filename}",
                    "metadata_path": str(metadata_path),
                    "original_width": original_image.width,
                    "original_height": original_image.height,
                    "restored_width": restored_image.width,
                    "restored_height": restored_image.height,
                    "faces_detected": face_count,
                    "faces_restored": len(face_images),
                    "individual_faces": face_images,
                    "processing_time": round(processing_time, 2),
                    "model_used": model_name,
                    "vram_used": vram_used,
                    "device": self.device,
                },
                "metadata": {
                    "task_type": "face_restore",
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                },
            }

        except Exception as e:
            logger.error(f"Failed to process restoration results: {e}")
            raise

    async def batch_restore_faces(
        self,
        images: List[Union[Image.Image, str, Path]],
        model_name: Optional[str] = None,
        upscale: int = 2,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Batch restore faces in multiple images"""
        try:
            start_time = time.time()
            logger.info(f"👤 Starting batch face restoration: {len(images)} images")

            results = []
            failed_images = []
            total_faces = 0

            for i, image in enumerate(images):
                try:
                    logger.info(f"Processing image {i+1}/{len(images)}")

                    result = await self.restore_faces(
                        image=image,
                        model_name=model_name,
                        upscale=upscale,
                        user_id=user_id,
                        **kwargs,
                    )

                    results.append(result)
                    total_faces += result["result"]["faces_detected"]

                except Exception as e:
                    logger.error(f"Failed to restore faces in image {i+1}: {e}")
                    failed_images.append({"index": i, "error": str(e)})

            total_time = time.time() - start_time

            return {
                "success": True,
                "batch_id": f"batch_face_restore_{int(time.time()*1000)}",
                "total_images": len(images),
                "successful_images": len(results),
                "failed_images": len(failed_images),
                "total_faces_detected": total_faces,
                "total_time": round(total_time, 2),
                "results": results,
                "failures": failed_images,
            }

        except Exception as e:
            logger.error(f"Batch face restoration failed: {e}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        avg_time = self.total_time / max(1, self.total_restorations)
        avg_faces = self.total_faces_processed / max(1, self.total_restorations)

        return {
            "service": "face_restore",
            "initialized": self.is_initialized,
            "current_model": self.current_model,
            "device": self.device,
            "available_models": list(self.available_models.keys()),
            "gfpgan_available": GFPGAN_AVAILABLE,
            "codeformer_available": CODEFORMER_AVAILABLE,
            "statistics": {
                "total_restorations": self.total_restorations,
                "total_faces_processed": self.total_faces_processed,
                "average_faces_per_image": round(avg_faces, 1),
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
            logger.info("🧹 Cleaning up Face Restore service...")

            if self.restorer is not None:
                del self.restorer
                self.restorer = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self.is_initialized = False
            self.current_model = None

            logger.info("✅ Face Restore service cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def switch_model(self, model_name: str):
        """Switch to a different face restoration model"""
        try:
            if self.current_model == model_name:
                logger.info(f"Model {model_name} already loaded")
                return

            logger.info(f"🔄 Switching from {self.current_model} to {model_name}")

            # Cleanup current model
            if self.restorer is not None:
                del self.restorer
                self.restorer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load new model
            await self._load_model(model_name)

            logger.info(f"✅ Successfully switched to {model_name}")

        except Exception as e:
            logger.error(f"Failed to switch to model {model_name}: {e}")
            raise


# Global service instance
_face_restore_service: Optional[FaceRestoreService] = None


async def get_face_restore_service() -> FaceRestoreService:
    """Get global face restore service instance (singleton pattern)"""
    global _face_restore_service
    if _face_restore_service is None:
        _face_restore_service = FaceRestoreService()
        await _face_restore_service.initialize()
    return _face_restore_service


async def cleanup_face_restore_service():
    """Cleanup global face restore service"""
    global _face_restore_service
    if _face_restore_service is not None:
        await _face_restore_service.cleanup()
        _face_restore_service = None
