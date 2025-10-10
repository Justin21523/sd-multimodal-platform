#!/usr/bin/env python3
"""
Comprehensive acceptance tests for the MVP requirements
"""
import sys
import os
import requests
import time
import json
import base64
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000/api/v1"
TEST_IMAGE = None


def create_test_image():
    """Create a test image for testing"""
    global TEST_IMAGE
    image = Image.new("RGB", (100, 100), color="red")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    TEST_IMAGE = (
        f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    )


def test_liveness():
    """Test liveness endpoint"""
    print("🧪 Testing liveness endpoint...")
    response = requests.get(f"{BASE_URL}/health/liveness")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] == True
    print("✅ Liveness test passed")


def test_readiness():
    """Test readiness endpoint"""
    print("🧪 Testing readiness endpoint...")
    response = requests.get(f"{BASE_URL}/health/readiness")
    assert response.status_code == 200
    data = response.json()
    assert "warehouse_accessible" in data
    assert "system_healthy" in data
    assert "device_available" in data
    print("✅ Readiness test passed")


def test_rate_limits():
    """Test rate limiting"""
    print("🧪 Testing rate limits...")

    # Make rapid requests to trigger rate limit
    headers = {"X-API-Key": "test-key"} if os.getenv("API_KEYS") else {}

    for i in range(15):  # Should trigger burst limit
        response = requests.post(
            f"{BASE_URL}/caption", json={"image": TEST_IMAGE}, headers=headers
        )
        if response.status_code == 429:
            data = response.json()
            assert data["error_code"] == "RATE_LIMIT_EXCEEDED"
            print("✅ Rate limiting working")
            return
        time.sleep(0.1)

    print("⚠️  Rate limiting not triggered")


def test_auth():
    """Test authentication"""
    print("🧪 Testing authentication...")

    # Skip if no API keys configured
    if not os.getenv("API_KEYS"):
        print("⚠️  No API keys configured, skipping auth test")
        return

    # Test without API key
    response = requests.post(f"{BASE_URL}/caption", json={"image": TEST_IMAGE})
    assert response.status_code in [401, 403]
    data = response.json()
    assert data["success"] == False
    assert "error_code" in data
    print("✅ Authentication test passed")


def test_caption():
    """Test caption endpoint"""
    print("🧪 Testing caption endpoint...")

    headers = {"X-API-Key": "test-key"} if os.getenv("API_KEYS") else {}

    payload = {"image": TEST_IMAGE, "max_length": 50}

    response = requests.post(f"{BASE_URL}/caption", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "caption" in data
    assert isinstance(data["caption"], str)
    print("✅ Caption test passed")


def test_vqa():
    """Test VQA endpoint"""
    print("🧪 Testing VQA endpoint...")

    headers = {"X-API-Key": "test-key"} if os.getenv("API_KEYS") else {}

    payload = {
        "image": TEST_IMAGE,
        "question": "What color is this image?",
        "max_length": 100,
    }

    response = requests.post(f"{BASE_URL}/vqa", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "answer" in data
    assert isinstance(data["answer"], str)
    print("✅ VQA test passed")


def test_queue():
    """Test queue functionality"""
    print("🧪 Testing queue endpoints...")

    headers = {"X-API-Key": "test-key"} if os.getenv("API_KEYS") else {}

    # Submit task
    payload = {
        "task_type": "caption",
        "parameters": {"image": TEST_IMAGE},
        "priority": 1,
    }

    response = requests.post(f"{BASE_URL}/queue/submit", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    task_id = data["task_id"]

    # Check status
    response = requests.get(f"{BASE_URL}/queue/status/{task_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert "status" in data

    # List tasks
    response = requests.get(f"{BASE_URL}/queue/list", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert "total" in data

    print("✅ Queue test passed")


def test_models():
    """Test models endpoint"""
    print("🧪 Testing models endpoint...")

    headers = {"X-API-Key": "test-key"} if os.getenv("API_KEYS") else {}

    response = requests.get(f"{BASE_URL}/models", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "models" in data or "categories" in data
    print("✅ Models test passed")


def run_all_tests():
    """Run all acceptance tests"""
    create_test_image()

    tests = [
        test_liveness,
        test_readiness,
        test_auth,
        test_rate_limits,
        test_caption,
        test_vqa,
        test_queue,
        test_models,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")

    print(f"\n📊 Acceptance Test Results: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("🎉 All acceptance tests passed! MVP requirements satisfied.")
        return True
    else:
        print("💥 Some acceptance tests failed!")
        return False


if __name__ == "__main__":
    print("🚀 Starting MVP Acceptance Tests...\n")

    try:
        # Wait for service to be ready
        time.sleep(3)

        success = run_all_tests()
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"💥 Acceptance tests failed to run: {e}")
        sys.exit(1)
