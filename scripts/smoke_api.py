#!/usr/bin/env python3
"""
Mock-safe API smoke tests.

Run against a local server started with:
  MOCK_GENERATION=true MINIMAL_MODE=true DEVICE=cpu uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from typing import Callable

import requests
from PIL import Image


API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
BASE_URL = f"{API_BASE}/api/v1"


def _assert_png_base64(value: str) -> None:
    if value.startswith("data:image/png;base64,"):
        value = value.split(",", 1)[1]
    assert value, "expected PNG base64"
    raw = base64.b64decode(value)
    assert raw.startswith(b"\x89PNG"), "decoded image is not PNG"


def _sample_image_data_url(color: str = "red") -> str:
    image = Image.new("RGB", (256, 256), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def test_health() -> None:
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"healthy", "degraded", "unhealthy"}
    assert data["service"]["name"] == "SD Multi-Modal Platform"


def test_models() -> None:
    response = requests.get(f"{BASE_URL}/models", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]["models"]) >= 1


def test_txt2img_mock() -> None:
    payload = {
        "prompt": "portfolio demo dashboard, crisp UI, cinematic lighting",
        "negative_prompt": "blurry, low quality",
        "width": 512,
        "height": 512,
        "steps": 4,
        "cfg_scale": 7.5,
        "seed": 21523,
        "num_images": 1,
        "save_images": False,
        "return_base64": True,
    }
    response = requests.post(f"{BASE_URL}/txt2img/", json=payload, timeout=20)
    assert response.status_code == 200, response.text[:500]
    data = response.json()
    assert data["success"] is True
    image = data["data"]["results"]["images"][0]
    _assert_png_base64(image["base64"])


def test_img2img_mock() -> None:
    payload = {
        "prompt": "turn this into a portfolio-ready generated image preview",
        "init_image": _sample_image_data_url("steelblue"),
        "width": 512,
        "height": 512,
        "strength": 0.55,
        "steps": 4,
        "cfg_scale": 7.0,
        "seed": 21523,
    }
    response = requests.post(f"{BASE_URL}/img2img/", json=payload, timeout=20)
    assert response.status_code == 200, response.text[:500]
    data = response.json()
    assert data["success"] is True
    assert data["data"]["images"], "expected saved image URL"


def test_queue_degrades_without_redis() -> None:
    response = requests.get(f"{BASE_URL}/queue/status", timeout=10)
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        data = response.json()
        assert "queue_stats" in data


def run_all_tests() -> bool:
    tests: list[Callable[[], None]] = [
        test_health,
        test_models,
        test_txt2img_mock,
        test_img2img_mock,
        test_queue_degrades_without_redis,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")

    print(f"\nSmoke results: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    print(f"Running API smoke tests against {BASE_URL}")
    time.sleep(1)
    sys.exit(0 if run_all_tests() else 1)
