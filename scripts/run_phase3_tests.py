#!/usr/bin/env python3
# scripts/run_phase3_tests.py
"""
Phase 3 Test Runner and Benchmark Script
Runs comprehensive tests for model management and generation functionality.
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
import requests
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings


def run_command(cmd, capture_output=True):
    """Run shell command and return result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True, cwd=project_root
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}")


def run_unit_tests():
    """Run unit tests for Phase 3 components."""
    print_section("UNIT TESTS")

    test_commands = [
        "python -m pytest tests/test_phase3_integration.py::TestModelRegistry -v",
        "python -m pytest tests/test_phase3_integration.py::TestModelManagerBasic -v",
        "python -m pytest tests/test_phase3_integration.py::TestTxt2ImgAPISchema -v",
    ]

    all_passed = True

    for cmd in test_commands:
        print(f"\n🧪 Running: {cmd}")
        success, stdout, stderr = run_command(cmd)

        if success:
            print("✅ PASSED")
            # Show key results
            lines = stdout.split("\n")
            for line in lines:
                if "passed" in line or "failed" in line or "error" in line:
                    print(f"   {line}")
        else:
            print("❌ FAILED")
            print(f"Error: {stderr}")
            all_passed = False

    return all_passed


def run_integration_tests():
    """Run integration tests (requires models)."""
    print_section("INTEGRATION TESTS")

    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists():
        print("⚠️  Models directory not found - skipping integration tests")
        print("   Run 'python scripts/install_models.py' to download models")
        return True

    cmd = "python -m pytest tests/test_phase3_integration.py::TestModelManagerIntegration -v -s"
    print(f"🧪 Running: {cmd}")

    success, stdout, stderr = run_command(cmd)

    if success:
        print("✅ INTEGRATION TESTS PASSED")
        return True
    else:
        print("❌ INTEGRATION TESTS FAILED")
        print(f"Error: {stderr}")
        return False


def run_api_tests():
    """Run API endpoint tests."""
    print_section("API TESTS")

    # Check if server is running
    try:
        response = requests.get(
            f"http://localhost:{settings.PORT}{settings.API_PREFIX}/health", timeout=5
        )
        server_running = response.status_code == 200
    except:
        server_running = False

    if not server_running:
        print("⚠️  Server not running - skipping API tests")
        print("   Start server with 'python scripts/start_phase3.py --start'")
        return True

    # Test basic endpoints
    base_url = f"http://localhost:{settings.PORT}{settings.API_PREFIX}"

    tests = [
        ("Health Check", f"{base_url}/health"),
        ("API Info", f"{base_url}/../info"),
        ("txt2img Status", f"{base_url}/txt2img/status"),
        ("Available Models", f"{base_url}/txt2img/models"),
    ]

    all_passed = True

    for test_name, url in tests:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {test_name}: OK ({response.status_code})")
            else:
                print(f"❌ {test_name}: Failed ({response.status_code})")
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")
            all_passed = False

    return all_passed


def run_generation_benchmark():
    """Run txt2img generation benchmark."""
    print_section("GENERATION BENCHMARK")

    # Check server availability
    base_url = f"http://localhost:{settings.PORT}{settings.API_PREFIX}"

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print("⚠️  Server not healthy - skipping benchmark")
            return True
    except:
        print("⚠️  Server not accessible - skipping benchmark")
        return True

    # Prepare test request
    test_request = {
        "prompt": "a red apple on white background, simple, clean",
        "negative_prompt": "blurry, complex, cluttered",
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,  # Fast generation for testing
        "guidance_scale": 7.5,
        "seed": 42,
        "num_images": 1,
        "save_images": False,  # Don't save for benchmark
        "return_base64": False,
    }

    print("🎨 Running generation benchmark...")
    print(f"   Prompt: {test_request['prompt']}")
    print(f"   Resolution: {test_request['width']}x{test_request['height']}")
    print(f"   Steps: {test_request['num_inference_steps']}")

    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/txt2img/", json=test_request, timeout=120)
        total_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                data = result["data"]["results"]
                print(f"✅ BENCHMARK SUCCESSFUL")
                print(f"   Total Time: {total_time:.2f}s")
                print(f"   Generation Time: {data['generation_time']:.2f}s")
                print(f"   VRAM Used: {data['vram_used_gb']:.2f}GB")
                print(f"   Model: {result['data']['model_used']['model_id']}")

                # Performance targets
                if total_time < 30:
                    print("🚀 Performance: EXCELLENT (< 30s)")
                elif total_time < 60:
                    print("⚡ Performance: GOOD (< 60s)")
                else:
                    print("🐌 Performance: SLOW (> 60s)")

                return True
            else:
                print(f"❌ Generation failed: {result.get('message', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Benchmark error: {e}")

    return False


def run_memory_stress_test():
    """Run memory stress test with multiple generations."""
    print_section("MEMORY STRESS TEST")

    base_url = f"http://localhost:{settings.PORT}{settings.API_PREFIX}"

    # Check server
    try:
        requests.get(f"{base_url}/health", timeout=5)
    except:
        print("⚠️  Server not accessible - skipping stress test")
        return True

    print("🔥 Running memory stress test (5 consecutive generations)...")

    stress_request = {
        "prompt": "memory stress test image",
        "width": 512,
        "height": 512,
        "num_inference_steps": 5,  # Minimal steps
        "num_images": 1,
        "save_images": False,
        "return_base64": False,
    }

    vram_usage = []
    generation_times = []

    for i in range(5):
        print(f"   Generation {i+1}/5...")

        try:
            response = requests.post(
                f"{base_url}/txt2img/", json={**stress_request, "seed": i}, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    data = result["data"]["results"]
                    vram_usage.append(data["vram_used_gb"])
                    generation_times.append(data["generation_time"])
                    print(
                        f"      ✅ Time: {data['generation_time']:.2f}s, VRAM: {data['vram_used_gb']:.2f}GB"
                    )
                else:
                    print(f"      ❌ Failed: {result.get('message', 'Unknown')}")
                    return False
            else:
                print(f"      ❌ HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"      ❌ Error: {e}")
            return False

    # Analyze results
    if vram_usage and generation_times:
        avg_vram = sum(vram_usage) / len(vram_usage)
        max_vram = max(vram_usage)
        avg_time = sum(generation_times) / len(generation_times)

        print(f"📊 STRESS TEST RESULTS:")
        print(f"   Average VRAM: {avg_vram:.2f}GB")
        print(f"   Peak VRAM: {max_vram:.2f}GB")
        print(f"   Average Time: {avg_time:.2f}s")
        print(f"   VRAM Stability: {max_vram - min(vram_usage):.2f}GB variation")

        # Check for memory leaks
        if max(vram_usage) - min(vram_usage) < 0.5:
            print("✅ Memory management: STABLE")
        else:
            print("⚠️  Memory management: POTENTIAL LEAK")

        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Phase 3 test runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run generation benchmark only"
    )
    parser.add_argument(
        "--stress", action="store_true", help="Run memory stress test only"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all tests and benchmarks"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick test suite")

    args = parser.parse_args()

    # Default to quick tests if no specific option
    if not any(
        [args.unit, args.integration, args.api, args.benchmark, args.stress, args.all]
    ):
        args.quick = True

    results = []

    print("🧪 SD Multi-Modal Platform - Phase 3 Test Suite")
    print(f"   Project root: {project_root}")
    print(f"   Device: {settings.DEVICE}")
    print(f"   Primary model: {settings.PRIMARY_MODEL}")

    if args.unit or args.all or args.quick:
        results.append(("Unit Tests", run_unit_tests()))

    if args.integration or args.all:
        results.append(("Integration Tests", run_integration_tests()))

    if args.api or args.all or args.quick:
        results.append(("API Tests", run_api_tests()))

    if args.benchmark or args.all:
        results.append(("Generation Benchmark", run_generation_benchmark()))

    if args.stress or args.all:
        results.append(("Memory Stress Test", run_memory_stress_test()))

    # Print summary
    print_section("TEST SUMMARY")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1

    print(f"\nTotal: {passed}/{total} test suites passed")

    if passed == total:
        print("🎉 All tests passed! Phase 3 is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
