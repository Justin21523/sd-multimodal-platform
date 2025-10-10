#!/usr/bin/env python3
"""
Comprehensive API smoke tests
"""
import sys
import os
import requests
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000/api/v1"


def test_health():
    """Test health endpoint"""
    print("🧪 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["cache_initialized"] == True
    assert "device_info" in data
    print("✅ Health test passed")


def test_caption():
    """Test caption endpoint"""
    print("🧪 Testing caption endpoint...")

    # Create a simple test image (red 100x100)
    from PIL import Image
    import io
    import base64

    image = Image.new("RGB", (100, 100), color="red")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    payload = {"image": f"data:image/jpeg;base64,{img_str}", "max_length": 50}

    response = requests.post(f"{BASE_URL}/caption", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "caption" in data
    print("✅ Caption test passed")


def test_queue():
    """Test queue endpoints"""
    print("🧪 Testing queue endpoints...")

    # Submit a task
    payload = {"task_type": "caption", "parameters": {"test": "data"}, "priority": 1}

    response = requests.post(f"{BASE_URL}/queue/submit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    task_id = data["task_id"]
    print(f"✅ Task submitted: {task_id}")

    # Check status
    response = requests.get(f"{BASE_URL}/queue/status/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    print("✅ Queue status test passed")

    # List tasks
    response = requests.get(f"{BASE_URL}/queue/list")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    print("✅ Queue list test passed")


def test_rate_limits():
    """Test rate limiting"""
    print("🧪 Testing rate limits...")

    # Make multiple rapid requests
    for i in range(5):
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 429:
            print("✅ Rate limiting working")
            return

    print("⚠️  Rate limiting not triggered (may be configured per minute)")


def run_all_tests():
    """Run all smoke tests"""
    tests = [test_health, test_caption, test_queue, test_rate_limits]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")

    print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("🚀 Starting Comprehensive API Smoke Tests...\n")

    try:
        # Wait for service to be ready
        time.sleep(2)

        success = run_all_tests()

        if success:
            print("🎉 All smoke tests passed!")
            sys.exit(0)
        else:
            print("💥 Some tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"💥 Smoke tests failed to run: {e}")
        sys.exit(1)
