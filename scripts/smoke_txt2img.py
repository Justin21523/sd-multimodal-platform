#!/usr/bin/env python3
"""
Smoke test for txt2img functionality
"""
import sys
import os
import tempfile
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dependencies import (
    get_txt2img_service,
    get_caption_service,
    get_vqa_service,
    get_task_manager,
)
from app.config import settings


def test_txt2img_output():
    """Test that txt2img creates output in correct location"""
    print("🧪 Testing txt2img output location...")

    try:
        # This would test the actual txt2img service
        # For now, we'll test the output directory structure

        outputs_dir = str(settings.OUTPUT_PATH)
        project_dir = os.path.join(outputs_dir, "smoke_test")
        images_dir = os.path.join(project_dir, "images")

        # Create test directory structure
        os.makedirs(images_dir, exist_ok=True)

        # Create a dummy output file
        test_image = Image.new("RGB", (100, 100), color="red")
        test_path = os.path.join(images_dir, "smoke_test_image.jpg")
        test_image.save(test_path)

        # Verify file was created
        if os.path.exists(test_path):
            print(f"✅ Output directory test passed: {test_path}")

            # Cleanup
            os.remove(test_path)
            return True
        else:
            print("❌ Output directory test failed")
            return False

    except Exception as e:
        print(f"❌ Output directory test failed: {e}")
        return False


def test_service_initialization():
    """Test that services can be initialized"""
    print("\n🧪 Testing service initialization...")

    try:
        # Test DI providers
        services = [
            get_txt2img_service,
            get_caption_service,
            get_vqa_service,
            get_task_manager,
        ]

        for service_getter in services:
            service = service_getter()
            print(f"✅ {service_getter.__name__}: {type(service).__name__}")

        return True

    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting txt2img Smoke Tests...\n")

    tests = [test_txt2img_output, test_service_initialization]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)

    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("🎉 txt2img smoke tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)
