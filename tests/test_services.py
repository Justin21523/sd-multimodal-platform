#!/usr/bin/env python3
# test_services.py - Test Phase 6 services

import pytest

# These are manual smoke checks that require local model weights and (optionally)
# a running Redis/Celery stack. Skip in CI/unit runs.
pytest.skip(
    "Service smoke tests require local models/redis; run manually if needed.",
    allow_module_level=True,
)

import asyncio
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
redis = pytest.importorskip("redis.asyncio", reason="redis package not installed")
from services.postprocess.face_restore_service import FaceRestoreService
from services.generation.img2img_service import Img2ImgService
from services.postprocess.upscale_service import UpscaleService

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.core.queue_manager import QueueManager
from app.config import settings


async def test_img2img_service():
    """Test Img2Img service"""
    print("🎨 Testing Img2Img service...")

    try:
        service = Img2ImgService()
        await service.initialize()

        # Create test image
        test_image = Image.new("RGB", (512, 512), color="blue")

        # Test generation
        result = await service.generate_image(
            prompt="a red apple",
            image=test_image,
            strength=0.8,
            num_inference_steps=1,  # Fast test
        )

        print(f"✅ Img2Img service test passed: {result['success']}")
        await service.cleanup()

        return True

    except Exception as e:
        print(f"❌ Img2Img service test failed: {e}")
        return False


async def test_upscale_service():
    """Test Upscale service"""
    print("🔍 Testing Upscale service...")

    try:
        service = UpscaleService()
        await service.initialize()

        # Create test image
        test_image = Image.new("RGB", (128, 128), color="green")

        # Test upscaling
        result = await service.upscale_image(image=test_image, scale=2)

        print(f"✅ Upscale service test passed: {result['success']}")
        await service.cleanup()

        return True

    except Exception as e:
        print(f"❌ Upscale service test failed: {e}")
        return False


async def test_face_restore_service():
    """Test Face Restore service"""
    print("👤 Testing Face Restore service...")

    try:
        service = FaceRestoreService()
        await service.initialize()

        # Create test image with simple face-like pattern
        test_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        test_image = Image.fromarray(test_img)

        # Test face restoration
        result = await service.restore_faces(image=test_image, upscale=2)

        print(f"✅ Face Restore service test passed: {result['success']}")
        await service.cleanup()

        return True

    except Exception as e:
        print(f"❌ Face Restore service test failed: {e}")
        return False


async def test_queue_manager():
    """Test Queue Manager"""
    print("📋 Testing Queue Manager...")

    try:
        queue_manager = QueueManager()
        await queue_manager.initialize()

        # Test enqueue
        task_id = await queue_manager.enqueue_task(
            task_type="txt2img",
            input_params={"prompt": "test", "steps": 20},
            user_id="test_user",
        )

        if task_id:
            print(f"✅ Task enqueued: {task_id}")

            # Test status check
            task_info = await queue_manager.get_task_status(task_id)
            if task_info:
                print(f"✅ Task status retrieved: {task_info.status}")

            # Test queue stats
            stats = await queue_manager.get_queue_status()
            print(f"✅ Queue stats: {stats.total_tasks} total tasks")

        await queue_manager.shutdown()

        return True

    except Exception as e:
        print(f"❌ Queue Manager test failed: {e}")
        return False


async def test_redis_connection():
    """Test Redis connection"""
    print("🔗 Testing Redis connection...")

    try:

        redis_client = redis.from_url(settings.get_redis_url())
        await redis_client.ping()
        await redis_client.close()

        print("✅ Redis connection test passed")
        return True

    except Exception as e:
        print(f"❌ Redis connection test failed: {e}")
        print(
            "💡 Make sure Redis is running: docker run -d --name redis -p 6379:6379 redis:7-alpine"
        )
        return False


async def test_imports():
    """Test all critical imports"""
    print("📦 Testing imports...")

    imports_to_test = [
        ("app.core.queue_manager", "QueueManager"),
        ("app.services.generation.img2img_service", "Img2ImgService"),
        ("app.services.postprocess.upscale_service", "UpscaleService"),
        ("app.services.postprocess.face_restore_service", "FaceRestoreService"),
        ("app.api.v1.queue", "router"),
        ("app.utils.middleware", "RequestLoggerMiddleware"),
    ]
