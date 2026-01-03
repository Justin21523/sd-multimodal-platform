#!/usr/bin/env python3
"""
Smoke test for cache and basic functionality
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.shared_cache import shared_cache
from app.config import settings
from services.models.model_cache import model_cache


def test_cache_setup():
    """Test cache directory setup"""
    print("🧪 Testing cache setup...")

    cache_info = shared_cache.get_cache_info()
    print(f"Cache root: {shared_cache.cache_root}")

    for key, path in cache_info.items():
        exists = os.path.exists(path)
        writable = os.access(path, os.W_OK) if exists else False
        status = "✅" if exists and writable else "❌"
        print(f"  {status} {key}: {path} (exists: {exists}, writable: {writable})")

    return all(
        os.path.exists(path) and os.access(path, os.W_OK)
        for path in cache_info.values()
    )


def test_device_setup():
    """Test device and precision setup"""
    print("\n🧪 Testing device setup...")

    print(f"Device: {model_cache.device}")
    print(f"Precision: {model_cache.dtype}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    return True


def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")

    print(f"App: {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Caption model: {settings.CAPTION_MODEL}")
    print(f"VQA model: {settings.VQA_MODEL}")
    print(f"Device: {settings.DEVICE}")
    print(f"Precision: {settings.PRECISION}")

    return True


if __name__ == "__main__":
    print("🚀 Starting SD Multimodal Platform Smoke Tests...\n")

    try:
        import torch

        tests = [test_cache_setup, test_device_setup, test_config_loading]

        results = []
        for test in tests:
            try:
                results.append(test())
            except Exception as e:
                print(f"❌ Test failed: {e}")
                results.append(False)

        print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")

        if all(results):
            print("🎉 All smoke tests passed!")
            sys.exit(0)
        else:
            print("💥 Some tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"💥 Failed to run smoke tests: {e}")
        sys.exit(1)
