# scripts/start_phase5.py
"""
Phase 5 startup script with queue system validation
"""

import asyncio
import subprocess
import sys
import time
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional

import redis
from celery import Celery

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from utils.logging_utils import setup_logging

setup_logging()


class Phase5Validator:
    """Validate Phase 5 queue system and post-processing"""

    def __init__(self):
        self.base_url = f"http://localhost:{settings.PORT}"
        self.api_prefix = settings.API_PREFIX

    def check_redis_connection(self) -> bool:
        """Check Redis connection"""
        print("🔍 Checking Redis connection...")

        try:
            r = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
            )

            # Test connection
            r.ping()

            # Test basic operations
            r.set("test_key", "test_value", ex=10)
            value = r.get("test_key")

            if value == "test_value":
                print("✅ Redis connection - OK")
                return True
            else:
                print("❌ Redis connection - Data integrity issue")
                return False

        except Exception as e:
            print(f"❌ Redis connection failed: {str(e)}")
            return False

    def check_celery_worker(self) -> bool:
        """Check if Celery worker is running"""
        print("🔍 Checking Celery worker...")

        try:
            celery_app = Celery(
                "sd_multimodal",
                broker=settings.CELERY_BROKER_URL,
                backend=settings.CELERY_RESULT_BACKEND,
            )

            # Get worker statistics
            inspect = celery_app.control.inspect()
            stats = inspect.stats()

            if stats:
                worker_count = len(stats)
                print(f"✅ Celery workers - {worker_count} active")

                # Show worker details
                for worker_name, worker_stats in stats.items():
                    print(f"   📋 Worker: {worker_name}")
                    print(
                        f"      Pool: {worker_stats.get('pool', {}).get('max-concurrency', 'N/A')}"
                    )

                return True
            else:
                print("❌ No Celery workers found")
                return False

        except Exception as e:
            print(f"❌ Celery worker check failed: {str(e)}")
            return False

    def check_postprocess_models(self) -> bool:
        """Check if post-processing models are available"""
        print("🔍 Checking post-processing models...")

        required_models = [
            (
                Path(settings.UPSCALE_MODELS_PATH) / "RealESRGAN_x4plus.pth",
                "Real-ESRGAN",
            ),
            (Path(settings.FACE_RESTORE_MODELS_PATH) / "GFPGANv1.4.pth", "GFPGAN"),
        ]

        all_available = True

        for model_path, model_name in required_models:
            if model_path.exists() and model_path.stat().st_size > 1024:
                size_mb = model_path.stat().st_size / 1024 / 1024
                print(f"✅ {model_name} - {size_mb:.1f}MB")
            else:
                print(f"❌ {model_name} - Missing or invalid")
                all_available = False

        return all_available

    def test_queue_api(self) -> bool:
        """Test queue API endpoints"""
        print("🔍 Testing queue API...")

        try:
            # Test queue stats endpoint
            response = requests.get(f"{self.base_url}{self.api_prefix}/queue/stats")

            if response.status_code == 200:
                stats = response.json()
                print("✅ Queue stats API - OK")
                print(
                    f"   📊 Redis connected: {stats['data'].get('redis_connected', False)}"
                )
                print(f"   📊 Active workers: {stats['data'].get('active_workers', 0)}")
                return True
            else:
                print(f"❌ Queue stats API failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Queue API test failed: {str(e)}")
            return False

    def test_queue_submission(self) -> Optional[str]:
        """Test task submission to queue"""
        print("🔍 Testing task submission...")

        try:
            # Submit a simple txt2img task
            task_data = {
                "prompt": "a cute cat",
                "width": 512,
                "height": 512,
                "num_inference_steps": 10,  # Short for testing
                "guidance_scale": 7.5,
            }

            response = requests.post(
                f"{self.base_url}{self.api_prefix}/queue/submit/txt2img", json=task_data
            )

            if response.status_code == 200:
                result = response.json()
                task_id = result["data"]["task_id"]
                print(f"✅ Task submission - OK (Task ID: {task_id})")
                return task_id
            else:
                print(f"❌ Task submission failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Task submission test failed: {str(e)}")
            return None

    def monitor_task_progress(self, task_id: str, timeout: int = 300) -> bool:
        """Monitor task progress until completion"""
        print(f"🔍 Monitoring task progress: {task_id}")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}{self.api_prefix}/queue/status/{task_id}"
                )

                if response.status_code == 200:
                    task_info = response.json()["data"]
                    status = task_info["status"]
                    progress = task_info.get("progress", 0)

                    print(f"   📊 Status: {status}, Progress: {progress:.1%}")

                    if status == "success":
                        print("✅ Task completed successfully")
                        return True
                    elif status == "failure":
                        error = task_info.get("error_message", "Unknown error")
                        print(f"❌ Task failed: {error}")
                        return False
                    elif status in ["pending", "started", "processing"]:
                        time.sleep(5)  # Wait 5 seconds before next check
                        continue
                    else:
                        print(f"❌ Unexpected task status: {status}")
                        return False

                else:
                    print(f"❌ Failed to get task status: {response.status_code}")
                    return False

            except Exception as e:
                print(f"❌ Error monitoring task: {str(e)}")
                return False

        print(f"❌ Task monitoring timed out after {timeout} seconds")
        return False

    def test_postprocess_pipeline(self) -> bool:
        """Test post-processing pipeline"""
        print("🔍 Testing post-processing pipeline...")

        try:
            # This would typically use an existing generated image
            # For now, we'll test the API endpoint structure

            postprocess_data = {
                "image_paths": ["test_image.png"],  # Placeholder
                "pipeline_type": "standard",
            }

            response = requests.post(
                f"{self.base_url}{self.api_prefix}/queue/submit/postprocess",
                json=postprocess_data,
            )

            # We expect this to fail due to missing image, but the API should respond properly
            if response.status_code in [200, 400, 422]:  # Valid API response
                print("✅ Post-processing API structure - OK")
                return True
            else:
                print(f"❌ Post-processing API failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Post-processing test failed: {str(e)}")
            return False

    def run_full_validation(self) -> bool:
        """Run complete Phase 5 validation"""
        print("🚀 Starting Phase 5 validation...\n")

        checks = [
            ("Redis Connection", self.check_redis_connection),
            ("Celery Workers", self.check_celery_worker),
            ("Post-processing Models", self.check_postprocess_models),
            ("Queue API", self.test_queue_api),
            ("Post-processing API", self.test_postprocess_pipeline),
        ]

        passed = 0
        total = len(checks)

        for check_name, check_func in checks:
            print(f"\n{'='*50}")
            print(f"Running: {check_name}")
            print("=" * 50)

            if check_func():
                passed += 1
            else:
                print(f"⚠️  {check_name} failed - check configuration")

        print(f"\n{'='*50}")
        print(f"VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Passed: {passed}/{total}")

        if passed == total:
            print("🎉 All checks passed! Phase 5 is ready.")

            # Optional: Run a full integration test
            print("\n🔄 Running integration test...")
            task_id = self.test_queue_submission()
            if task_id:
                success = self.monitor_task_progress(task_id, timeout=120)
                if success:
                    print("🎉 Integration test passed!")
                else:
                    print("⚠️  Integration test failed")

            return True
        else:
            print("❌ Some checks failed. Please fix issues before proceeding.")
            return False


def start_redis_server():
    """Start Redis server if not running"""
    print("🔍 Checking if Redis server is running...")

    try:
        r = redis.Redis(host="localhost", port=6379, db=0)
        r.ping()
        print("✅ Redis server is already running")
        return True
    except:
        print("📦 Starting Redis server...")
        try:
            subprocess.Popen(
                ["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(3)  # Give Redis time to start

            # Test again
            r = redis.Redis(host="localhost", port=6379, db=0)
            r.ping()
            print("✅ Redis server started successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to start Redis server: {str(e)}")
            print("   Please install and start Redis manually:")
            print(
                "   - Ubuntu/Debian: sudo apt install redis-server && sudo systemctl start redis"
            )
            print("   - macOS: brew install redis && brew services start redis")
            print("   - Windows: Download from https://redis.io/download")
            return False


def start_celery_worker():
    """Start Celery worker if not running"""
    print("🔍 Starting Celery worker...")

    try:
        # Start Celery worker in background
        cmd = [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "services.queue.tasks",
            "worker",
            "--loglevel=info",
            "--concurrency=1",
        ]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=project_root
        )

        print("✅ Celery worker started in background")
        print(f"   PID: {process.pid}")
        print("   Logs: Check celery worker output for details")

        return True

    except Exception as e:
        print(f"❌ Failed to start Celery worker: {str(e)}")
        return False


def main():
    """Main Phase 5 startup function"""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5 startup and validation")
    parser.add_argument(
        "--validate-only", action="store_true", help="Only run validation"
    )
    parser.add_argument("--skip-redis", action="store_true", help="Skip Redis startup")
    parser.add_argument(
        "--skip-celery", action="store_true", help="Skip Celery startup"
    )
    parser.add_argument(
        "--install-models",
        action="store_true",
        help="Install post-processing models first",
    )

    args = parser.parse_args()

    print("🚀 SD Multi-Modal Platform - Phase 5 Startup")
    print("=" * 60)

    # Install models if requested
    if args.install_models:
        print("\n📦 Installing post-processing models...")
        try:
            from scripts.install_postprocess_models import PostprocessModelInstaller

            installer = PostprocessModelInstaller()
            if not installer.install_all():
                print("❌ Model installation failed")
                return False
        except Exception as e:
            print(f"❌ Model installation error: {str(e)}")
            return False

    # Start services if not in validate-only mode
    if not args.validate_only:
        # Start Redis
        if not args.skip_redis:
            if not start_redis_server():
                return False

        # Start Celery worker
        if not args.skip_celery:
            if not start_celery_worker():
                return False

        # Wait for services to be ready
        print("\n⏳ Waiting for services to initialize...")
        time.sleep(5)

    # Run validation
    validator = Phase5Validator()
    success = validator.run_full_validation()

    if success:
        print("\n🎉 Phase 5 startup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Access the API documentation: http://localhost:8000/docs")
        print(
            "2. Monitor Celery tasks: Start Flower with 'celery -A services.queue.tasks flower'"
        )
        print("3. Submit tasks via queue API endpoints")
        print("4. Check queue status at: http://localhost:8000/api/v1/queue/stats")

        # Show useful commands
        print("\n🔧 Useful commands:")
        print("  Start main app: uvicorn app.main:app --reload")
        print("  Monitor Celery: celery -A services.queue.tasks events")
        print("  Redis CLI: redis-cli")
        print("  View queue: celery -A services.queue.tasks inspect active")

    else:
        print("\n❌ Phase 5 startup failed!")
        print("Please check the error messages above and fix any issues.")
        return False

    return success


if __name__ == "__main__":
    success = main()
