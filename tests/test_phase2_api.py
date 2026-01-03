# tests/test_phase2_api.py
"""
API and middleware smoke tests.

NOTE: This repo cannot reliably use Starlette/FastAPI TestClient in this runtime
(threaded anyio portal hang). These tests use `httpx.AsyncClient` via the
fixture in `tests/conftest.py`.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from app.config import settings
from tests.conftest import assert_request_headers, assert_response_structure


@pytest.mark.api
@pytest.mark.health
class TestHealthEndpoints:
    async def test_health_check_basic(self, client, mock_torch):
        response = await client.get(f"{settings.API_PREFIX}/health")
        assert response.status_code == 200
        data = response.json()

        required_fields = ["status", "timestamp", "service", "system"]
        assert_response_structure(data, required_fields)
        assert_request_headers(response)

        service = data["service"]
        assert service["name"] == "SD Multi-Modal Platform"
        assert service["version"] == "1.0.0-phase2"
        assert service["phase"] == "Phase 2: Backend Framework & Basic API Services"

        system = data["system"]
        assert "platform" in system
        assert "python_version" in system
        assert "device" in system
        assert "cuda_available" in system

        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    async def test_simple_health_check(self, client):
        response = await client.get(f"{settings.API_PREFIX}/health/simple")
        assert response.status_code == 200
        data = response.json()

        required_fields = ["status", "service", "timestamp"]
        assert_response_structure(data, required_fields)
        assert_request_headers(response)

        assert data["status"] == "ok"
        assert data["service"] == "sd-multimodal-platform"

    async def test_detailed_health_check(self, client, mock_torch):
        response = await client.get(f"{settings.API_PREFIX}/health/detailed")
        assert response.status_code == 200
        data = response.json()

        basic_fields = ["status", "timestamp", "service", "system"]
        detailed_fields = ["configuration", "directory_status"]
        assert_response_structure(data, basic_fields + detailed_fields)

        config = data["configuration"]
        assert "environment" in config
        assert "model_paths" in config

        model_paths = config["model_paths"]
        assert "sd_model_path" in model_paths
        assert "sdxl_model_path" in model_paths
        assert "controlnet_path" in model_paths

        dir_status = data["directory_status"]
        assert "sd_model_path" in dir_status
        assert "sdxl_model_path" in dir_status

        for dir_info in dir_status.values():
            assert "exists" in dir_info
            assert "is_directory" in dir_info
            assert "path" in dir_info

    async def test_health_check_no_cuda(self, client, mock_torch_no_cuda):
        response = await client.get(f"{settings.API_PREFIX}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["system"]["cuda_available"] is False

    async def test_health_check_gpu_info_error(self, client, mock_torch):
        with patch(
            "torch.cuda.get_device_name", side_effect=RuntimeError("GPU access failed")
        ):
            response = await client.get(f"{settings.API_PREFIX}/health")
            assert response.status_code == 200
            data = response.json()
            if "gpu" in data:
                assert "error" in data["gpu"] or data["status"] == "degraded"


@pytest.mark.api
@pytest.mark.middleware
class TestMiddleware:
    async def test_request_id_header(self, client):
        response = await client.get(f"{settings.API_PREFIX}/health")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

    async def test_process_time_header(self, client):
        response = await client.get(f"{settings.API_PREFIX}/health")
        assert "X-Process-Time" in response.headers
        assert response.headers["X-Process-Time"].endswith("s")

    async def test_cors_options(self, client):
        response = await client.options(
            f"{settings.API_PREFIX}/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code in {200, 204}


@pytest.mark.api
class TestErrorHandling:
    async def test_404_is_json(self, client):
        response = await client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert_request_headers(response)

    async def test_405_is_json(self, client):
        response = await client.post(f"{settings.API_PREFIX}/health")  # GET-only
        assert response.status_code == 405
        data = response.json()
        assert "detail" in data
        assert_request_headers(response)


@pytest.mark.api
class TestRootEndpoint:
    async def test_root_endpoint(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()

        assert data["name"] == settings.APP_NAME
        assert data["version"] == settings.APP_VERSION
        assert data["docs"] == f"{settings.API_PREFIX}/docs"
        assert data["openapi"] == f"{settings.API_PREFIX}/openapi.json"


@pytest.mark.api
class TestLoggingIntegration:
    async def test_logging_includes_request_id(self, client, caplog):
        caplog.set_level("INFO")
        caplog.clear()
        response = await client.get(f"{settings.API_PREFIX}/health")
        assert response.status_code == 200

        request_logs = [
            record for record in caplog.records if hasattr(record, "request_id")
        ]
        assert len(request_logs) > 0
        assert len(getattr(request_logs[0], "request_id")) == 8

    async def test_request_logging_messages(self, client, caplog):
        caplog.set_level("INFO")
        caplog.clear()
        response = await client.get(f"{settings.API_PREFIX}/health")
        assert response.status_code == 200

        messages = [record.message for record in caplog.records]
        assert any("Request start" in msg for msg in messages)
        assert any("Request end" in msg for msg in messages)


@pytest.mark.api
@pytest.mark.performance
class TestPerformanceAndLoad:
    async def test_health_check_performance(self, client):
        start_time = time.time()
        response = await client.get(f"{settings.API_PREFIX}/health")
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 1.0

        process_time_header = response.headers.get("X-Process-Time", "0s")
        assert process_time_header.endswith("s")

    async def test_concurrent_requests(self, client):
        async def make_request():
            return await client.get(f"{settings.API_PREFIX}/health")

        responses = await asyncio.gather(*[make_request() for _ in range(10)])
        assert all(r.status_code == 200 for r in responses)

        request_ids = [r.headers.get("X-Request-ID") for r in responses]
        assert len(set(request_ids)) == len(request_ids)
