#!/usr/bin/env python3
# scripts/start_phase2.py
"""
Phase 2 startup and validation script.
Validates environment, starts the server, and runs basic acceptance tests.
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import torch


def check_environment() -> Dict[str, Any]:
    """Check if the environment is properly configured."""
    print("🔍 Checking environment configuration...")

    issues = []
    warnings = []

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(
            f"Python version {python_version.major}.{python_version.minor} is too old. Need 3.8+"
        )
    elif python_version < (3, 10):
        warnings.append(
            f"Python {python_version.major}.{python_version.minor} works but 3.10+ is recommended"
        )

    # Check required packages
    required_packages = ["fastapi", "uvicorn", "torch", "pydantic"]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        issues.append(f"Missing required packages: {', '.join(missing_packages)}")

    # Check CUDA availability if configured
    try:
        cuda_available = torch.cuda.is_available()
        device_setting = os.getenv("DEVICE", "cuda")

        if device_setting == "cuda" and not cuda_available:
            warnings.append("DEVICE=cuda but CUDA not available. Will fallback to CPU.")
        elif cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"  ✅ CUDA available: {gpu_name}")
    except ImportError:
        issues.append("PyTorch not installed")

    # Check directories
    required_dirs = ["models", "outputs", "logs", "assets"]

    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            warnings.append(
                f"Directory {dir_name} doesn't exist. Will be created automatically."
            )

    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            warnings.append(
                ".env file not found. Copy .env.example to .env and configure."
            )
        else:
            issues.append("No .env or .env.example file found.")

    return {
        "issues": issues,
        "warnings": warnings,
        "cuda_available": cuda_available if "torch" in sys.modules else False,
    }


def start_server(
    port: int = 8000, background: bool = False
) -> Optional[subprocess.Popen]:
    """Start the FastAPI server."""
    print(f"🚀 Starting FastAPI server on port {port}...")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--reload" if not background else "--no-reload",
    ]

    if background:
        # Start in background for testing
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait a moment for startup
        print("  ⏳ Waiting for server startup...")
        time.sleep(3)

        return process
    else:
        # Start in foreground
        print("  📝 Starting in foreground mode (Ctrl+C to stop)")
        print("  🌐 API docs will be available at: http://localhost:8000/api/v1/docs")
        subprocess.run(cmd)
        return None


def validate_api(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Validate API endpoints are working correctly."""
    print("🧪 Running API validation tests...")

    results = {"tests_passed": 0, "tests_failed": 0, "test_results": []}

    tests = [
        {
            "name": "Root Endpoint",
            "url": f"{base_url}/",
            "method": "GET",
            "expected_status": 200,
            "required_fields": ["service", "version", "docs", "health"],
        },
        {
            "name": "Health Check",
            "url": f"{base_url}/api/v1/health",
            "method": "GET",
            "expected_status": 200,
            "required_fields": ["status", "timestamp", "service", "system"],
        },
        {
            "name": "Simple Health Check",
            "url": f"{base_url}/api/v1/health/simple",
            "method": "GET",
            "expected_status": 200,
            "required_fields": ["status", "service", "timestamp"],
        },
        {
            "name": "Detailed Health Check",
            "url": f"{base_url}/api/v1/health/detailed",
            "method": "GET",
            "expected_status": 200,
            "required_fields": ["configuration", "directory_status"],
        },
        {
            "name": "API Documentation",
            "url": f"{base_url}/api/v1/docs",
            "method": "GET",
            "expected_status": 200,
            "content_type_check": "text/html",
        },
        {
            "name": "OpenAPI Schema",
            "url": f"{base_url}/api/v1/openapi.json",
            "method": "GET",
            "expected_status": 200,
            "content_type_check": "application/json",
        },
    ]

    for test in tests:
        try:
            print(f"  🔄 Testing: {test['name']}")

            response = requests.get(test["url"], timeout=10)

            # Check status code
            if response.status_code != test["expected_status"]:
                raise AssertionError(
                    f"Expected status {test['expected_status']}, got {response.status_code}"
                )

            # Check content type if specified
            if "content_type_check" in test:
                content_type = response.headers.get("content-type", "")
                if test["content_type_check"] not in content_type:
                    raise AssertionError(
                        f"Expected content-type to contain {test['content_type_check']}"
                    )

            # Check required fields for JSON responses
            if "required_fields" in test:
                try:
                    data = response.json()
                    for field in test["required_fields"]:
                        if field not in data:
                            raise AssertionError(
                                f"Required field '{field}' not found in response"
                            )
                except json.JSONDecodeError:
                    raise AssertionError("Response is not valid JSON")

            # Check headers
            headers = response.headers
            if "X-Request-ID" not in headers:
                raise AssertionError("X-Request-ID header missing")

            if "X-Process-Time" not in headers:
                raise AssertionError("X-Process-Time header missing")

            print(f"    ✅ {test['name']} - PASSED")
            results["tests_passed"] += 1
            results["test_results"].append(
                {
                    "name": test["name"],
                    "status": "PASSED",
                    "response_time": response.elapsed.total_seconds(),
                }
            )

        except Exception as e:
            print(f"    ❌ {test['name']} - FAILED: {str(e)}")
            results["tests_failed"] += 1
            results["test_results"].append(
                {"name": test["name"], "status": "FAILED", "error": str(e)}
            )

    return results


def run_performance_test(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Run basic performance tests."""
    print("⚡ Running performance tests...")

    health_url = f"{base_url}/api/v1/health"
    times = []

    # Warm up
    requests.get(health_url)

    # Run test
    for i in range(20):
        start = time.time()
        response = requests.get(health_url)
        end = time.time()

        if response.status_code == 200:
            times.append(end - start)
        else:
            print(f"  ⚠️  Request {i+1} failed with status {response.status_code}")

    if times:
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        p95_time = sorted(times)[int(0.95 * len(times))]

        print(f"  📊 Performance Results:")
        print(f"    Average: {avg_time:.3f}s")
        print(f"    Min: {min_time:.3f}s")
        print(f"    Max: {max_time:.3f}s")
        print(f"    95th percentile: {p95_time:.3f}s")

        # Performance criteria
        if avg_time < 0.1:
            print("    ✅ Performance: EXCELLENT (avg < 100ms)")
        elif avg_time < 0.2:
            print("    ✅ Performance: GOOD (avg < 200ms)")
        elif avg_time < 0.5:
            print("    ⚠️  Performance: ACCEPTABLE (avg < 500ms)")
        else:
            print("    ❌ Performance: POOR (avg > 500ms)")

        return {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "p95_time": p95_time,
            "total_requests": len(times),
        }
    else:
        print("  ❌ No successful requests for performance measurement")
        return {"error": "No successful requests"}


def main():
    """Main function to run Phase 2 validation."""
    parser = argparse.ArgumentParser(description="Phase 2 Startup & Validation Script")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--test-only", action="store_true", help="Only run tests, don't start server"
    )
    parser.add_argument(
        "--start-only", action="store_true", help="Only start server, don't run tests"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Start server in background for testing",
    )

    args = parser.parse_args()

    print("🎯 SD Multi-Modal Platform - Phase 2 Validation")
    print("=" * 60)

    # Check environment
    env_check = check_environment()

    if env_check["issues"]:
        print("❌ Environment Issues Found:")
        for issue in env_check["issues"]:
            print(f"  - {issue}")
        print("\nPlease fix these issues before continuing.")
        return 1

    if env_check["warnings"]:
        print("⚠️  Environment Warnings:")
        for warning in env_check["warnings"]:
            print(f"  - {warning}")
        print()

    server_process = None

    try:
        # Start server if requested
        if not args.test_only:
            server_process = start_server(args.port, args.background)

            if args.background:
                # Wait for server to be ready
                print("  ⏳ Waiting for server to be ready...")
                max_retries = 30
                for i in range(max_retries):
                    try:
                        response = requests.get(
                            f"http://localhost:{args.port}/api/v1/health/simple",
                            timeout=1,
                        )
                        if response.status_code == 200:
                            print("  ✅ Server is ready!")
                            break
                    except:
                        pass
                    time.sleep(1)
                else:
                    print("  ❌ Server failed to start within 30 seconds")
                    return 1

        # Run tests if requested
        if not args.start_only:
            print()
            base_url = f"http://localhost:{args.port}"

            # API validation
            api_results = validate_api(base_url)

            print()
            print(f"📋 API Validation Results:")
            print(f"  ✅ Tests Passed: {api_results['tests_passed']}")
            print(f"  ❌ Tests Failed: {api_results['tests_failed']}")

            if api_results["tests_failed"] > 0:
                print("\n❌ Some tests failed. Check the output above for details.")
                return 1

            # Performance test
            print()
            perf_results = run_performance_test(base_url)

            print()
            print("🎉 Phase 2 validation completed successfully!")
            print("✅ Backend Framework & Basic API Services are working correctly.")
            print()
            print("Next steps:")
            print("  1. Check API documentation at: http://localhost:8000/api/v1/docs")
            print("  2. Proceed to Phase 3: Model Management & First Model Integration")

            return 0

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1

    finally:
        # Clean up background server
        if server_process and args.background:
            print("\n🧹 Cleaning up background server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()
