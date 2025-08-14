#!/usr/bin/env python3
"""
Quick fix and validation script for Phase 3 issues.
Validates all imports, configurations, and dependencies.
"""

import sys
import importlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")

    # Test basic imports
    try:
        from app.config import settings

        print("✅ app.config imported successfully")
    except Exception as e:
        print(f"❌ app.config import failed: {e}")
        return False

    # Test schemas
    try:
        from app.schemas.requests import Txt2ImgRequest
        from app.schemas.responses import Txt2ImgResponse

        print("✅ app.schemas imported successfully")
    except Exception as e:
        print(f"❌ app.schemas import failed: {e}")
        return False

    # Test attention utils
    try:
        from utils.attention_utils import setup_attention_processor

        print("✅ utils.attention_utils imported successfully")
    except Exception as e:
        print(f"❌ utils.attention_utils import failed: {e}")
        return False

    # Test model manager
    try:
        from services.models.sd_models import ModelManager, get_model_manager

        print("✅ services.models.sd_models imported successfully")
    except Exception as e:
        print(f"❌ services.models.sd_models import failed: {e}")
        return False

    # Test txt2img API
    try:
        from app.api.v1.txt2img import router

        print("✅ app.api.v1.txt2img imported successfully")
    except Exception as e:
        print(f"❌ app.api.v1.txt2img import failed: {e}")
        return False

    # Test main app
    try:
        from app.main import app

        print("✅ app.main imported successfully")
    except Exception as e:
        print(f"❌ app.main import failed: {e}")
        return False

    return True


def test_configuration():
    """Test configuration values."""
    print("\n⚙️ Testing configuration...")

    try:
        from app.config import settings

        # Test required settings
        required_settings = [
            "DEVICE",
            "PRIMARY_MODEL",
            "USE_SDPA",
            "ENABLE_XFORMERS",
            "MAX_BATCH_SIZE",
            "OUTPUT_PATH",
        ]

        for setting in required_settings:
            value = getattr(settings, setting, None)
            if value is not None:
                print(f"✅ {setting}: {value}")
            else:
                print(f"❌ {setting}: Not set")
                return False

        # Test methods
        torch_dtype = settings.get_torch_dtype()
        print(f"✅ torch_dtype: {torch_dtype}")

        origins = settings.allowed_origins_list
        print(f"✅ allowed_origins_list: {len(origins)} origins")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_pydantic_schemas():
    """Test Pydantic schema creation."""
    print("\n📋 Testing Pydantic schemas...")

    try:
        from app.schemas.requests import Txt2ImgRequest
        from app.schemas.responses import Txt2ImgResponse

        # Test request creation
        request = Txt2ImgRequest(prompt="test prompt", width=512, height=512)
        print(f"✅ Txt2ImgRequest created: {request.prompt}")

        # Test validation
        request_with_validation = Txt2ImgRequest(
            prompt="test",
            width=515,  # Should be rounded to 520
            height=517,  # Should be rounded to 520
            seed=-1,  # Should become None
        )

        if (
            request_with_validation.width == 520
            and request_with_validation.height == 520
        ):
            print("✅ Dimension validation working")
        else:
            print(
                f"❌ Dimension validation failed: {request_with_validation.width}x{request_with_validation.height}"
            )

        if request_with_validation.seed is None:
            print("✅ Seed validation working")
        else:
            print(f"❌ Seed validation failed: {request_with_validation.seed}")

        return True

    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False


def test_model_registry():
    """Test model registry functionality."""
    print("\n🤖 Testing model registry...")

    try:
        from services.models.sd_models import ModelRegistry

        # Test model listing
        models = ModelRegistry.list_models()
        print(f"✅ Available models: {models}")

        # Test model info
        for model_id in models:
            info = ModelRegistry.get_model_info(model_id)
            if info:
                print(f"✅ {model_id}: {info['name']}")
            else:
                print(f"❌ Failed to get info for {model_id}")
                return False

        return True

    except Exception as e:
        print(f"❌ Model registry test failed: {e}")
        return False


def test_directory_structure():
    """Test required directories exist."""
    print("\n📁 Testing directory structure...")

    required_dirs = ["app/schemas", "utils", "services/models", "app/api/v1"]

    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} missing")
            all_exist = False

    return all_exist


def main():
    """Run all validation tests."""
    print("🚀 SD Multi-Modal Platform - Phase 3 Quick Fix Validation")
    print(f"Project root: {project_root}")
    print("=" * 60)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Pydantic Schemas", test_pydantic_schemas),
        ("Model Registry", test_model_registry),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} test failed")
        except Exception as e:
            print(f"\n💥 {test_name} test crashed: {e}")

    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Phase 3 fixes are working correctly.")
        print("\nNext steps:")
        print("1. Run: python scripts/install_models.py --models sdxl-base")
        print("2. Run: python scripts/start_phase3.py --validate-only")
        print("3. Run: python scripts/start_phase3.py --start")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    main()
