# scripts/quick_fix_phase5.py
"""
Quick fix and validation script for Phase 5 configuration issues
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_config_import():
    """Test if config can be imported without errors"""
    print("🔍 Testing configuration import...")

    try:
        from app.config import settings

        print("✅ Config import successful")

        # Test key attributes
        required_attrs = [
            "MODELS_PATH",
            "ENABLE_QUEUE",
            "ENVIRONMENT",
            "MINIMAL_MODE",
            "UPSCALE_MODELS_PATH",
            "FACE_RESTORE_MODELS_PATH",
            "REDIS_HOST",
            "REDIS_PORT",
            "CELERY_BROKER_URL",
        ]

        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(settings, attr):
                missing_attrs.append(attr)

        if missing_attrs:
            print(f"❌ Missing configuration attributes: {missing_attrs}")
            return False
        else:
            print("✅ All required configuration attributes present")

        # Test computed properties
        try:
            origins = settings.allowed_origins_list
            models_path = settings.models_path_obj
            redis_url = settings.redis_url
            print("✅ Computed properties working")
        except Exception as e:
            print(f"❌ Computed properties error: {str(e)}")
            return False

        return True

    except Exception as e:
        print(f"❌ Config import failed: {str(e)}")
        return False


def test_task_manager_import():
    """Test if task manager can be imported"""
    print("\n🔍 Testing task manager import...")

    try:
        from services.queue.task_manager import get_task_manager, TaskStatus, TaskInfo

        print("✅ Task manager import successful")

        # Test TaskStatus enum
        try:
            status = TaskStatus.PENDING
            print(f"✅ TaskStatus enum working: {status}")
        except Exception as e:
            print(f"❌ TaskStatus enum error: {str(e)}")
            return False

        # Test TaskInfo dataclass
        try:
            from datetime import datetime

            task_info = TaskInfo(
                task_id="test",
                task_type="txt2img",
                status=TaskStatus.PENDING,
                progress=0.0,
                created_at=datetime.utcnow(),
            )
            task_dict = task_info.to_dict()
            print("✅ TaskInfo dataclass working")
        except Exception as e:
            print(f"❌ TaskInfo dataclass error: {str(e)}")
            return False

        # Test task manager initialization
        try:
            task_manager = get_task_manager()
            print("✅ Task manager initialization successful")

            # Test Redis availability (won't fail if Redis is not running)
            if hasattr(task_manager, "redis_client") and task_manager.redis_client:
                print("✅ Redis client initialized")
            else:
                print("⚠️  Redis not available - fallback mode active")

        except Exception as e:
            print(f"❌ Task manager initialization error: {str(e)}")
            return False

        return True

    except Exception as e:
        print(f"❌ Task manager import failed: {str(e)}")
        return False


def test_pipeline_manager_import():
    """Test if pipeline manager can be imported"""
    print("\n🔍 Testing pipeline manager import...")

    try:
        from services.postprocess.pipeline_manager import (
            get_pipeline_manager,
            PostprocessPipeline,
        )

        print("✅ Pipeline manager import successful")

        # Test pipeline creation
        try:
            pipeline_manager = get_pipeline_manager()
            print("✅ Pipeline manager initialization successful")
        except Exception as e:
            print(f"❌ Pipeline manager initialization error: {str(e)}")
            return False

        return True

    except Exception as e:
        print(f"❌ Pipeline manager import failed: {str(e)}")
        return False


def test_main_app_import():
    """Test if main app can be imported"""
    print("\n🔍 Testing main app import...")

    try:
        from app.main import app

        print("✅ Main app import successful")
        return True

    except Exception as e:
        print(f"❌ Main app import failed: {str(e)}")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔍 Checking dependencies...")

    required_packages = [
        ("redis", "Redis client"),
        ("celery", "Celery task queue"),
        ("fastapi", "FastAPI framework"),
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers library"),
        ("PIL", "Pillow image library"),
        ("pydantic", "Pydantic validation"),
        ("pydantic_settings", "Pydantic settings"),
    ]

    missing_packages = []

    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {description} - installed")
        except ImportError:
            print(f"❌ {description} - missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True


def create_env_file():
    """Create .env file if it doesn't exist"""
    print("\n🔍 Checking .env file...")

    env_file = project_root / ".env"
    env_example = project_root / ".env.example"

    if env_file.exists():
        print("✅ .env file exists")
        return True

    if env_example.exists():
        try:
            import shutil

            shutil.copy(env_example, env_file)
            print("✅ Created .env from .env.example")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {str(e)}")
            return False
    else:
        print("⚠️  .env.example not found, creating basic .env")

        basic_env_content = """# Basic configuration for Phase 5
ENVIRONMENT=development
MINIMAL_MODE=false
DEVICE=auto
ENABLE_QUEUE=true
REDIS_HOST=localhost
REDIS_PORT=6379
MODELS_PATH=./models
OUTPUT_PATH=./outputs
"""

        try:
            with open(env_file, "w") as f:
                f.write(basic_env_content)
            print("✅ Created basic .env file")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {str(e)}")
            return False


def check_directories():
    """Check and create required directories"""
    print("\n🔍 Checking directories...")

    from app.config import settings

    required_dirs = [
        settings.MODELS_PATH,
        settings.OUTPUT_PATH,
        settings.UPSCALE_MODELS_PATH,
        settings.FACE_RESTORE_MODELS_PATH,
    ]

    for directory in required_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"✅ {directory} - exists")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ {directory} - created")
            except Exception as e:
                print(f"❌ {directory} - failed to create: {str(e)}")
                return False

    return True


def main():
    """Main fix and validation function"""
    print("🚀 Phase 5 Configuration Fix & Validation")
    print("=" * 60)

    checks = [
        ("Dependencies", check_dependencies),
        ("Environment File", create_env_file),
        ("Configuration Import", test_config_import),
        ("Directories", check_directories),
        ("Task Manager", test_task_manager_import),
        ("Pipeline Manager", test_pipeline_manager_import),
        ("Main App", test_main_app_import),
    ]

    passed = 0
    total = len(checks)

    for check_name, check_func in checks:
        print(f"\n{'='*40}")
        print(f"Running: {check_name}")
        print("=" * 40)

        if check_func():
            passed += 1
            print(f"✅ {check_name} - PASSED")
        else:
            print(f"❌ {check_name} - FAILED")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All checks passed! Phase 5 configuration is ready.")
        print("\n📋 Next steps:")
        print("1. Start Redis: redis-server")
        print(
            "2. Start Celery worker: celery -A services.queue.tasks worker --loglevel=info"
        )
        print("3. Start the application: python scripts/start_phase5.py")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.")

        if passed >= total * 0.7:  # 70% success rate
            print("\n💡 Most checks passed. You may be able to run in minimal mode:")
            print("   MINIMAL_MODE=true python app/main.py")

        return False


if __name__ == "__main__":
    success = main()
