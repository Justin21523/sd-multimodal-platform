#!/usr/bin/env python3
"""
Debug script for Phase 3 startup issues.
Helps identify why the FastAPI application fails to start properly.
"""
import logging
import sys
import asyncio
import time
import subprocess
import psutil
from pathlib import Path
import requests
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from utils.logging_utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


class StartupDebugger:
    """Debug Phase 3 startup issues step by step."""

    def __init__(self):
        self.issues_found = []
        self.warnings = []

    def debug_step(self, step_name: str):
        """Decorator for debug steps."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                print(f"\n🔍 {step_name}")
                print("-" * 50)
                try:
                    result = func(*args, **kwargs)
                    if result:
                        print(f"✅ {step_name}: PASSED")
                    else:
                        print(f"❌ {step_name}: FAILED")
                        self.issues_found.append(step_name)
                    return result
                except Exception as e:
                    print(f"💥 {step_name}: ERROR - {e}")
                    self.issues_found.append(f"{step_name}: {e}")
                    return False

            return wrapper

        return decorator

    def check_system_requirements(self):
        """Check basic system requirements."""
        print(f"Python version: {sys.version}")
        print(f"Working directory: {Path.cwd()}")
        print(f"Project root: {project_root}")

        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name}")
            print(f"Total VRAM: {total_vram:.1f}GB")

        # Check device setting
        device_setting = settings.DEVICE
        print(f"Device setting: {device_setting}")

        if device_setting == "cuda" and not cuda_available:
            self.issues_found.append("CUDA requested but not available")
            return False

        return True

    def check_configuration(self):
        """Validate configuration settings."""
        try:
            print(f"API_PREFIX: {settings.API_PREFIX}")
            print(f"HOST: {settings.HOST}")
            print(f"PORT: {settings.PORT}")
            print(f"PRIMARY_MODEL: {settings.PRIMARY_MODEL}")
            print(f"OUTPUT_PATH: {settings.OUTPUT_PATH}")

            # Test directory creation
            settings.ensure_directories()
            print("✅ Directories created successfully")

            # Test torch dtype
            torch_dtype = settings.get_torch_dtype()
            print(f"Torch dtype: {torch_dtype}")

            return True

        except Exception as e:
            print(f"Configuration error: {e}")
            return False

    def check_model_files(self):
        """Check if required model files exist."""
        models_base = Path("models")

        if not models_base.exists():
            print("❌ Models directory doesn't exist")
            print("Run: python scripts/install_models.py --models sdxl-base")
            return False

        # Check primary model
        from services.models.sd_models import ModelRegistry

        primary_model = settings.PRIMARY_MODEL
        model_info = ModelRegistry.get_model_info(primary_model)

        if not model_info:
            print(f"❌ Unknown primary model: {primary_model}")
            return False

        model_path = models_base / model_info["local_path"]
        print(f"Checking model path: {model_path}")

        if not model_path.exists():
            print(f"❌ Model files not found at: {model_path}")
            print("Run: python scripts/install_models.py --models sdxl-base")
            return False

        # Check key files
        key_files = [
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ]

        missing_files = []
        for key_file in key_files:
            file_path = model_path / key_file
            if not file_path.exists():
                missing_files.append(key_file)

        if missing_files:
            print(f"❌ Missing key files: {missing_files}")
            return False

        print(f"✅ Model files complete for {primary_model}")
        return True

    def test_model_manager(self):
        """Test model manager initialization."""
        try:
            from services.models.sd_models import get_model_manager

            print("Creating model manager...")
            manager = get_model_manager()

            print("Attempting initialization...")

            # Run initialization
            async def init_test():
                return await manager.initialize()

            success = asyncio.run(init_test())

            if success:
                print(f"✅ Model manager initialized with: {manager.current_model_id}")
                print(f"Startup time: {manager.startup_time:.2f}s")

                # Test status
                status = manager.get_status()
                print(f"Manager status: {status['is_initialized']}")

                # Cleanup
                async def cleanup_test():
                    await manager.cleanup()

                asyncio.run(cleanup_test())
                return True
            else:
                print("❌ Model manager initialization failed")
                return False

        except Exception as e:
            print(f"Model manager error: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_app_import(self):
        """Test FastAPI application import."""
        try:
            from app.main import app

            print(f"✅ FastAPI app imported: {type(app)}")
            print(f"App title: {app.title}")
            print(f"App version: {app.version}")

            # Check routes
            route_count = len(app.routes)
            print(f"Registered routes: {route_count}")

            return True

        except Exception as e:
            print(f"App import error: {e}")
            import traceback

            traceback.print_exc()
            return False

    def check_port_availability(self):
        """Check if the configured port is available."""
        import socket

        host = settings.HOST
        port = settings.PORT

        print(f"Checking port {port} availability...")

        # Simple socket test first
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                print(f"⚠️  Port {port} is already in use")
                self.warnings.append(f"Port {port} in use")

                # Try to find which process is using the port
                try:
                    import subprocess

                    result = subprocess.run(
                        ["lsof", "-i", f":{port}"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.stdout:
                        print(f"Process using port {port}:")
                        print(result.stdout)
                except:
                    print("Could not identify process using the port")

                return True  # Not a fatal error, just a warning
            else:
                print(f"✅ Port {port} is available")
                return True

        except Exception as e:
            print(f"❌ Port check failed: {e}")
            return False

    def test_manual_server_start(self):
        """Test starting the server manually."""
        print("Attempting manual server start...")

        try:
            import uvicorn
            from app.main import app

            # Start server in a separate process
            import multiprocessing
            import time

            def run_server():
                uvicorn.run(
                    app,
                    host=settings.HOST,
                    port=settings.PORT,
                    log_level="info",
                    access_log=False,
                )

            # Start server process
            server_process = multiprocessing.Process(target=run_server)
            server_process.start()

            # Wait a bit for startup
            time.sleep(10)

            # Test if server responds
            try:
                response = requests.get(
                    f"http://localhost:{settings.PORT}/health", timeout=5
                )
                if response.status_code == 200:
                    print("✅ Server responds to health check")
                    server_process.terminate()
                    return True
                else:
                    print(f"❌ Server responded with status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Server not responding: {e}")

            # Cleanup
            server_process.terminate()
            server_process.join(timeout=5)

            return False

        except Exception as e:
            print(f"Manual server start error: {e}")
            return False

    def run_full_diagnosis(self):
        """Run complete diagnosis."""
        print("🚨 SD Multi-Modal Platform - Startup Diagnosis")
        print("=" * 60)

        # Run all diagnostic steps
        steps = [
            self.check_system_requirements,
            self.check_configuration,
            self.check_model_files,
            self.test_model_manager,
            self.test_app_import,
            self.check_port_availability,
            # self.test_manual_server_start  # Skip this for now as it's complex
        ]

        passed = 0
        for step in steps:
            if step():
                passed += 1

        print("\n" + "=" * 60)
        print("🎯 DIAGNOSIS SUMMARY")
        print("=" * 60)

        print(f"Tests passed: {passed}/{len(steps)}")

        if self.issues_found:
            print(f"\n❌ ISSUES FOUND ({len(self.issues_found)}):")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"   {i}. {issue}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")

        # Provide specific recommendations
        print(f"\n💡 RECOMMENDATIONS:")

        if "Model Files Check" in [issue.split(":")[0] for issue in self.issues_found]:
            print(
                "   1. Download models: python scripts/install_models.py --models sdxl-base"
            )

        if "Model Manager Initialization" in [
            issue.split(":")[0] for issue in self.issues_found
        ]:
            print(
                "   2. Check VRAM availability - may need to reduce model size or use CPU"
            )
            print("   3. Try setting ENABLE_CPU_OFFLOAD=true in .env")

        if "CUDA requested but not available" in self.issues_found:
            print("   4. Set DEVICE=cpu in .env file for CPU-only mode")

        if any("Port" in issue for issue in self.issues_found):
            print("   5. Change PORT in .env or stop conflicting services")

        print("   6. Try running with minimal config:")
        print(
            "      DEVICE=cpu ENABLE_CPU_OFFLOAD=true python scripts/start_phase3.py --start"
        )

        return len(self.issues_found) == 0


def main():
    """Main diagnosis function."""
    debugger = StartupDebugger()
    success = debugger.run_full_diagnosis()

    if success:
        print("\n🎉 No critical issues found! Try starting the server again.")
    else:
        print("\n🔧 Please fix the issues above before starting the server.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
