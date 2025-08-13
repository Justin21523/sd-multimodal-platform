#!/usr/bin/env python3
"""
Simplified debug script for Phase 3 startup issues.
Focuses on the most critical checks without complex dependencies.
"""

import sys
import os
import time
from pathlib import Path
import socket

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_imports():
    """Check if all required modules can be imported."""
    print("🔍 Testing critical imports...")

    try:
        from app.config import settings

        print(
            f"✅ Config loaded: DEVICE={settings.DEVICE}, PRIMARY_MODEL={settings.PRIMARY_MODEL}"
        )
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

    try:
        from app.schemas.requests import Txt2ImgRequest

        print("✅ Schemas imported successfully")
    except Exception as e:
        print(f"❌ Schemas import failed: {e}")
        return False

    try:
        from utils.attention_utils import setup_attention_processor

        print("✅ Attention utils imported")
    except Exception as e:
        print(f"❌ Attention utils import failed: {e}")
        return False

    try:
        from services.models.sd_models import ModelRegistry

        models = ModelRegistry.list_models()
        print(f"✅ Model registry: {models}")
    except Exception as e:
        print(f"❌ Model registry failed: {e}")
        return False

    return True


def check_torch():
    """Check PyTorch and CUDA availability."""
    print("\n🔍 Checking PyTorch...")

    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name} ({total_vram:.1f}GB)")

        return True

    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
        return False


def check_models():
    """Check if model files exist."""
    print("\n🔍 Checking model files...")

    try:
        from app.config import settings
        from services.models.sd_models import ModelRegistry

        models_base = Path("models")
        if not models_base.exists():
            print("❌ Models directory doesn't exist")
            print("📝 Solution: python scripts/install_models.py --models sdxl-base")
            return False

        primary_model = settings.PRIMARY_MODEL
        model_info = ModelRegistry.get_model_info(primary_model)
        model_path = models_base / model_info["local_path"]  # type: ignore

        if not model_path.exists():
            print(f"❌ Model files not found: {model_path}")
            print("📝 Solution: python scripts/install_models.py --models sdxl-base")
            return False

        # Check key files
        key_files = ["model_index.json", "unet", "vae"]
        missing = []
        for key_file in key_files:
            if not (model_path / key_file).exists():
                missing.append(key_file)

        if missing:
            print(f"❌ Missing model components: {missing}")
            return False

        print(f"✅ Model files complete: {primary_model}")
        return True

    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False


def check_port():
    """Check if port is available."""
    print("\n🔍 Checking port availability...")

    try:
        from app.config import settings

        host = settings.HOST
        port = settings.PORT

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            print(f"⚠️  Port {port} is already in use")
            print("📝 Solution: Change PORT in .env or stop conflicting service")
            return True  # Warning, not error
        else:
            print(f"✅ Port {port} is available")
            return True

    except Exception as e:
        print(f"❌ Port check failed: {e}")
        return False


def test_fastapi_import():
    """Test FastAPI app import."""
    print("\n🔍 Testing FastAPI app import...")

    try:
        # Set minimal mode to avoid model initialization
        os.environ["MINIMAL_MODE"] = "true"

        from app.main import app

        print(f"✅ FastAPI app imported: {app.title}")
        print(f"Routes registered: {len(app.routes)}")
        return True

    except Exception as e:
        print(f"❌ FastAPI import failed: {e}")
        return False


def test_model_manager():
    """Test model manager initialization."""
    print("\n🔍 Testing model manager...")

    try:
        import asyncio
        from services.models.sd_models import get_model_manager

        async def test_init():
            manager = get_model_manager()
            success = await manager.initialize()
            if success:
                print(f"✅ Model manager initialized: {manager.current_model_id}")
                await manager.cleanup()
                return True
            else:
                print("❌ Model manager initialization failed")
                return False

        return asyncio.run(test_init())

    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        print(f"Error details: {type(e).__name__}: {e}")
        return False


def main():
    """Run simplified diagnosis."""
    print("🚨 SD Multi-Modal Platform - Simple Diagnosis")
    print("=" * 50)

    tests = [
        ("Imports", check_imports),
        ("PyTorch", check_torch),
        ("Model Files", check_models),
        ("Port", check_port),
        ("FastAPI Import", test_fastapi_import),
    ]

    # Add model manager test only if basic checks pass
    basic_passed = 0
    for test_name, test_func in tests:
        print()
        if test_func():
            basic_passed += 1
        else:
            print(f"\n💡 Fix the {test_name} issue before proceeding")

    print(f"\n📊 Basic checks: {basic_passed}/{len(tests)} passed")

    # Only test model manager if basic checks pass
    if basic_passed == len(tests):
        print("\n🔬 Running advanced tests...")
        if test_model_manager():
            print("\n🎉 All tests passed! System is ready.")
            print("\n📝 Next steps:")
            print("1. python scripts/start_phase3.py --start")
            return True
        else:
            print("\n⚠️  Model manager failed - try solutions below")

    print("\n🛠️  QUICK SOLUTIONS:")
    print("1. Download models: python scripts/install_models.py --models sdxl-base")
    print("2. Use CPU mode: export DEVICE=cpu")
    print("3. Enable memory optimization: export ENABLE_CPU_OFFLOAD=true")
    print("4. Try minimal mode: MINIMAL_MODE=true python app/main.py")

    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
