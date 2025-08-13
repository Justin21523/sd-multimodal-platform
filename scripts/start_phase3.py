#!/usr/bin/env python3
# scripts/start_phase3.py
"""
Phase 3 Startup and Validation Script
Validates system requirements, models, and starts the application.
"""

import logging
import asyncio
import sys
import time
from pathlib import Path
import requests
import subprocess
import signal
from typing import Optional, Dict, Any
import torch
import torch.version
import diffusers
import fastapi
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from services.models.sd_models import ModelRegistry
from utils.logging_utils import setup_logging, get_request_logger

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


class Phase3Validator:
    """Validates Phase 3 requirements and system readiness."""

    def __init__(self):
        self.base_models_path = Path(settings.OUTPUT_PATH).parent / "models"
        self.validation_results = {}

    async def check_models_availability(self) -> Dict[str, Any]:
        """Check if required models are downloaded and accessible."""
        logger.info("Checking model availability...")

        model_status = {}
        available_models = []

        for model_id in ModelRegistry.list_models():
            model_info = ModelRegistry.get_model_info(model_id)
            model_path = self.base_models_path / model_info["local_path"]  # type: ignore

            is_available = model_path.exists()
            if is_available:
                # Check for key files
                key_files = [
                    "model_index.json",
                    "unet/diffusion_pytorch_model.safetensors",
                ]
                all_files_exist = all((model_path / f).exists() for f in key_files)
                is_available = all_files_exist

            model_status[model_id] = {
                "available": is_available,
                "path": str(model_path),
                "info": model_info,
            }

            if is_available:
                available_models.append(model_id)

        results = {
            "total_models": len(ModelRegistry.list_models()),
            "available_models": len(available_models),
            "model_status": model_status,
            "available_model_ids": available_models,
            "primary_model_available": settings.PRIMARY_MODEL in available_models,
        }

        logger.info(
            f"Models check: {len(available_models)}/{len(ModelRegistry.list_models())} available"
        )

        if not available_models:
            logger.warning(
                "❌ No models available! Run 'python scripts/install_models.py' first"
            )
        elif not results["primary_model_available"]:
            logger.warning(f"⚠️  Primary model '{settings.PRIMARY_MODEL}' not available")
            logger.info(f"Available models: {available_models}")

        return results

    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements and hardware."""
        logger.info("Checking system requirements...")

        try:
            # Check CUDA
            cuda_available = torch.cuda.is_available()
            gpu_info = {}

            if cuda_available:
                gpu_info = {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_count": torch.cuda.device_count(),
                    "total_vram_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                    ),
                    "cuda_version": torch.version.cuda,
                }

            system_info = {
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "diffusers_version": diffusers.__version__,
                "fastapi_version": fastapi.__version__,
                "cuda_available": cuda_available,
                "device_setting": settings.DEVICE,
                "gpu_info": gpu_info,
            }

            # Check device compatibility
            device_compatible = True
            warnings = []

            if settings.DEVICE == "cuda" and not cuda_available:
                device_compatible = False
                warnings.append("CUDA device requested but not available")

            if cuda_available and gpu_info["total_vram_gb"] < 4:
                warnings.append("GPU has less than 4GB VRAM - may cause issues")

            system_info.update(
                {"device_compatible": device_compatible, "warnings": warnings}
            )

            logger.info(
                f"System check: Python {system_info['python_version']}, PyTorch {system_info['torch_version']}"
            )
            if cuda_available:
                logger.info(
                    f"GPU: {gpu_info['gpu_name']} ({gpu_info['total_vram_gb']}GB VRAM)"
                )
            else:
                logger.info("Running in CPU mode")

            return system_info

        except ImportError as e:
            logger.error(f"❌ Missing required dependency: {e}")
            return {"error": f"Import error: {e}"}

    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate application configuration."""
        logger.info("Validating configuration...")

        config_status = {
            "settings_loaded": True,
            "directories_created": False,
            "env_file_exists": False,
            "issues": [],
        }

        try:
            # Check .env file
            env_file = Path.cwd() / ".env"
            config_status["env_file_exists"] = env_file.exists()

            if not env_file.exists():
                config_status["issues"].append("No .env file found (using defaults)")

            # Try to create directories
            settings.ensure_directories()
            config_status["directories_created"] = True

            # Validate critical settings
            if not settings.SECRET_KEY or settings.SECRET_KEY == "your-secret-key-here":
                config_status["issues"].append("SECRET_KEY not set or using default")

            if settings.MAX_BATCH_SIZE > 4:
                config_status["issues"].append(
                    "MAX_BATCH_SIZE > 4 may cause memory issues"
                )

        except Exception as e:
            config_status["issues"].append(f"Configuration error: {e}")

        logger.info(f"Configuration check: {len(config_status['issues'])} issues found")
        return config_status

    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete system validation."""
        logger.info("🔍 Running Phase 3 validation...")

        results = {
            "timestamp": time.time(),
            "phase": "Phase 3: Model Management & Real Generation",
            "overall_status": "unknown",
        }

        # Run all checks
        results["system"] = self.check_system_requirements()
        results["configuration"] = await self.validate_configuration()
        results["models"] = await self.check_models_availability()

        # Determine overall status
        has_errors = False
        has_warnings = False

        # Check for critical errors
        if "error" in results["system"]:
            has_errors = True
        elif not results["system"].get("device_compatible", True):
            has_errors = True
        elif not results["models"]["available_models"]:
            has_errors = True

        # Check for warnings
        if results["system"].get("warnings"):
            has_warnings = True
        if results["configuration"].get("issues"):
            has_warnings = True
        if not results["models"]["primary_model_available"]:
            has_warnings = True

        if has_errors:
            results["overall_status"] = "error"
        elif has_warnings:
            results["overall_status"] = "warning"
        else:
            results["overall_status"] = "ready"

        return results

    def print_validation_summary(self, results: Dict[str, Any]) -> None:
        """Print human-readable validation summary."""
        status = results["overall_status"]

        print("\n" + "=" * 80)
        print("SD MULTI-MODAL PLATFORM - PHASE 3 VALIDATION RESULTS")
        print("=" * 80)

        # Overall status
        status_emoji = {"ready": "✅", "warning": "⚠️", "error": "❌"}
        print(f"\nOverall Status: {status_emoji.get(status, '❓')} {status.upper()}")

        # System info
        print(f"\n🖥️  SYSTEM INFORMATION")
        sys_info = results["system"]
        if "error" not in sys_info:
            print(f"   Python: {sys_info['python_version']}")
            print(f"   PyTorch: {sys_info['torch_version']}")
            print(f"   Device: {sys_info['device_setting']}")
            print(f"   CUDA Available: {sys_info['cuda_available']}")

            if sys_info["cuda_available"]:
                gpu_info = sys_info["gpu_info"]
                print(f"   GPU: {gpu_info['gpu_name']} ({gpu_info['total_vram_gb']}GB)")

            if sys_info.get("warnings"):
                for warning in sys_info["warnings"]:
                    print(f"   ⚠️  {warning}")
        else:
            print(f"   ❌ {sys_info['error']}")

        # Model status
        print(f"\n🤖 MODEL AVAILABILITY")
        models = results["models"]
        print(
            f"   Available: {models['available_models']}/{models['total_models']} models"
        )
        print(f"   Primary Model Ready: {models['primary_model_available']}")

        for model_id, status in models["model_status"].items():
            status_icon = "✅" if status["available"] else "❌"
            print(f"   {status_icon} {model_id}: {status['info']['name']}")

        # Configuration issues
        config = results["configuration"]
        if config.get("issues"):
            print(f"\n⚙️  CONFIGURATION ISSUES")
            for issue in config["issues"]:
                print(f"   • {issue}")

        # Next steps
        print(f"\n📋 NEXT STEPS")
        if status == "error":
            print("   ❌ Cannot start application due to critical errors")
            if not results["models"]["available_models"]:
                print("   1. Run: python scripts/install_models.py")
            if "error" in results["system"]:
                print("   2. Fix system requirements (see above)")
        elif status == "warning":
            print("   ⚠️  Application can start but has warnings")
            print("   1. Review warnings above")
            print("   2. Start application: python scripts/start_phase3.py --start")
        else:
            print("   ✅ Ready to start application")
            print("   1. Start: python scripts/start_phase3.py --start")

        print("=" * 80)


class Phase3Launcher:
    """Handles application startup and monitoring."""

    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{settings.HOST}:{settings.PORT}"

    async def wait_for_server_ready(self, timeout: int = 60) -> bool:
        """Wait for server to become ready."""
        logger.info(f"Waiting for server at {self.base_url}...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}{settings.API_PREFIX}/health", timeout=5
                )
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") in ["healthy", "degraded"]:
                        logger.info("✅ Server is ready!")
                        return True
            except requests.exceptions.RequestException:
                pass

            await asyncio.sleep(2)

        logger.error(f"❌ Server not ready after {timeout}s")
        return False

    async def test_txt2img_endpoint(self) -> bool:
        """Test txt2img functionality with a simple request."""
        logger.info("Testing txt2img endpoint...")

        try:
            test_request = {
                "prompt": "a simple red apple on white background",
                "negative_prompt": "blurry, low quality",
                "width": 512,
                "height": 512,
                "num_inference_steps": 10,
                "seed": 42,
                "save_images": True,
                "return_base64": False,
            }

            response = requests.post(
                f"{self.base_url}{settings.API_PREFIX}/txt2img/",
                json=test_request,
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info("✅ txt2img test successful!")
                    logger.info(
                        f"Generated {result['data']['results']['num_images']} images"
                    )
                    logger.info(
                        f"Generation time: {result['data']['results']['generation_time']}s"
                    )
                    return True
                else:
                    logger.error(
                        f"❌ txt2img test failed: {result.get('message', 'Unknown error')}"
                    )
            else:
                logger.error(
                    f"❌ txt2img test failed with status {response.status_code}"
                )

        except Exception as e:
            logger.error(f"❌ txt2img test error: {e}")

        return False

    def start_server(self) -> bool:
        """Start the FastAPI server."""
        logger.info("Starting FastAPI server...")

        try:
            # Start server process
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                settings.HOST,
                "--port",
                str(settings.PORT),
                "--reload",
            ]

            self.server_process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            logger.info(f"Server started with PID: {self.server_process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the server gracefully."""
        if self.server_process:
            logger.info("Stopping server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, forcing...")
                self.server_process.kill()
            finally:
                self.server_process = None

    async def launch_and_test(self) -> bool:
        """Launch server and run integration tests."""
        logger.info("🚀 Launching Phase 3 application...")

        # Start server
        if not self.start_server():
            return False

        try:
            # Wait for server to be ready
            server_ready = await self.wait_for_server_ready()
            if not server_ready:
                return False

            # Test basic functionality
            txt2img_works = await self.test_txt2img_endpoint()

            if txt2img_works:
                logger.info("🎉 Phase 3 application is fully operational!")
                print(f"\n✅ Server running at: {self.base_url}")
                print(f"📚 API docs: {self.base_url}{settings.API_PREFIX}/docs")
                print(f"❤️  Health check: {self.base_url}{settings.API_PREFIX}/health")
                print(
                    f"🎨 Generate image: {self.base_url}{settings.API_PREFIX}/txt2img/"
                )
                print("\nPress Ctrl+C to stop the server")

                # Keep server running
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Received shutdown signal")

                return True
            else:
                logger.error("❌ txt2img endpoint not working properly")
                return False

        finally:
            self.stop_server()


async def main():
    """Main Phase 3 startup script."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 startup and validation")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, don't start server",
    )
    parser.add_argument(
        "--start", action="store_true", help="Start server after validation"
    )
    parser.add_argument(
        "--force", action="store_true", help="Start server even with warnings"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test txt2img endpoint (server must be running)",
    )

    args = parser.parse_args()

    # Test existing server
    if args.test_only:
        launcher = Phase3Launcher()
        success = await launcher.test_txt2img_endpoint()
        sys.exit(0 if success else 1)

    # Run validation
    validator = Phase3Validator()
    validation_results = await validator.run_full_validation()
    validator.print_validation_summary(validation_results)

    status = validation_results["overall_status"]

    # Exit if only validating
    if args.validate_only:
        sys.exit(0 if status != "error" else 1)

    # Check if we should start
    should_start = args.start or (status == "ready")

    if status == "error" and not args.force:
        logger.error("❌ Cannot start due to critical errors. Use --force to override.")
        sys.exit(1)
    elif status == "warning" and not (args.start or args.force):
        logger.warning("⚠️  System has warnings. Use --start to proceed anyway.")
        sys.exit(1)
    elif should_start or args.force:
        # Launch application
        launcher = Phase3Launcher()
        success = await launcher.launch_and_test()
        sys.exit(0 if success else 1)
    else:
        logger.info("Use --start to launch the application")
        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
