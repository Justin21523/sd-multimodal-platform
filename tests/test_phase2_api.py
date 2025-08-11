# tests/test_phase2_api.py
"""
Test suite for Phase 2: Backend Framework & Basic API Services
Tests API endpoints, middleware, error handling, and health checks.
"""
import pytest
import json
import time
import concurrent.futures
import threading
from unittest.mock import patch, MagicMock

from conftest import (
    assert_response_structure,
    assert_request_headers,
    assert_error_response_format,
    PerformanceTimer,
)
from app.config import settings


@pytest.mark.api
@pytest.mark.health
class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check_basic(self, client, mock_torch):
        """Test basic health check endpoint."""
        response = client.get(f"{settings.API_PREFIX}/health")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        required_fields = ["status", "timestamp", "service", "system"]
        assert_response_structure(data, required_fields)
        assert_request_headers(response)

        # Check service info
        service = data["service"]
        assert service["name"] == "SD Multi-Modal Platform"
        assert service["version"] == "1.0.0-phase2"
        assert service["phase"] == "Phase 2: Backend Framework & Basic API Services"

        # Check system info
        system = data["system"]
        assert "platform" in system
        assert "python_version" in system
        assert "device" in system
        assert "cuda_available" in system

        # Check status is valid
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_simple_health_check(self, client):
        """Test simple health check for load balancers."""
        response = client.get(f"{settings.API_PREFIX}/health/simple")

        assert response.status_code == 200
        data = response.json()

        required_fields = ["status", "service", "timestamp"]
        assert_response_structure(data, required_fields)
        assert_request_headers(response)

        assert data["status"] == "ok"
        assert data["service"] == "sd-multimodal-platform"
        assert isinstance(data["timestamp"], str)

    def test_detailed_health_check(self, client, mock_torch):
        """Test detailed health check with configuration info."""
        response = client.get(f"{settings.API_PREFIX}/health/detailed")

        assert response.status_code == 200
        data = response.json()

        # Should include all basic health info plus configuration
        basic_fields = ["status", "timestamp", "service", "system"]
        detailed_fields = ["configuration", "directory_status"]
        assert_response_structure(data, basic_fields + detailed_fields)

        # Check configuration sections
        config = data["configuration"]
        assert "environment" in config
        assert "model_paths" in config

        # Check environment config
        env_config = config["environment"]
        assert "device" in env_config
        assert "torch_dtype" in env_config
        assert "max_workers" in env_config

        # Check model paths
        model_paths = config["model_paths"]
        assert "sd_model_path" in model_paths
        assert "sdxl_model_path" in model_paths

        # Check directory status
        dir_status = data["directory_status"]
        assert "sd_model_path" in dir_status
        assert "sdxl_model_path" in dir_status

        # Each directory should have exists and is_directory fields
        for dir_info in dir_status.values():
            assert "exists" in dir_info
            assert "is_directory" in dir_info
            assert "path" in dir_info

    def test_health_check_no_cuda(self, client, mock_torch_no_cuda):
        """Test health check when CUDA is not available."""
        response = client.get(f"{settings.API_PREFIX}/health")

        assert response.status_code == 200
        data = response.json()

        # Should still be healthy but with degraded status if device was cuda
        assert data["system"]["cuda_available"] is False

        # If device setting is cuda but cuda not available, should be degraded
        if settings.DEVICE == "cuda":
            assert data["status"] == "degraded"
            assert any(
                "CUDA not available" in issue for issue in data.get("issues", [])
            )

    def test_health_check_gpu_info_error(self, client):
        """Test health check when GPU info retrieval fails."""
        with patch(
            "torch.cuda.get_device_name", side_effect=RuntimeError("GPU access failed")
        ):
            response = client.get(f"{settings.API_PREFIX}/health")

            # Should still return 200 but with error info
            assert response.status_code == 200
            data = response.json()

            # Status might be degraded due to GPU info failure
            if "gpu" in data:
                assert "error" in data["gpu"] or data["status"] == "degraded"


class TestMiddleware:
    """Test middleware functionality."""

    def test_request_id_middleware(self, client):
        """Test that request ID is added to responses."""
        response = client.get(f"{settings.API_PREFIX}/health")

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8  # Short UUID

    def test_process_time_middleware(self, client):
        """Test that process time is added to responses."""
        response = client.get(f"{settings.API_PREFIX}/health")

        assert "X-Process-Time" in response.headers
        process_time = response.headers["X-Process-Time"]
        assert process_time.endswith("s")
        assert float(process_time[:-1]) >= 0

    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        # OPTIONS request to check CORS
        response = client.options(
            f"{settings.API_PREFIX}/health", headers={"Origin": "http://localhost:3000"}
        )

        # Should not be 405 Method Not Allowed due to CORS middleware
        assert response.status_code != 405


class TestErrorHandling:
    """Test error handling and exception responses."""

    def test_404_error_format(self, client):
        """Test 404 error returns structured format."""
        response = client.get("/nonexistent-endpoint")

        assert response.status_code == 404
        data = response.json()

        # Check error structure
        assert data["success"] is False
        assert "error" in data
        assert "request_id" in data
        assert "timestamp" in data

    def test_method_not_allowed_format(self, client):
        """Test 405 error returns structured format."""
        response = client.post(f"{settings.API_PREFIX}/health")  # GET-only endpoint

        assert response.status_code == 405
        data = response.json()

        assert data["success"] is False
        assert "error" in data
        assert "request_id" in data

    @patch("app.api.v1.health.torch.cuda.get_device_name")
    def test_internal_error_handling(self, mock_gpu_info, client):
        """Test internal error handling."""
        # Make GPU info raise an exception
        mock_gpu_info.side_effect = RuntimeError("GPU access failed")

        response = client.get(f"{settings.API_PREFIX}/health")

        # Should still return 200 but with error info in GPU section
        assert response.status_code == 200
        data = response.json()

        # GPU info should contain error
        if "gpu" in data:
            assert "error" in data["gpu"] or data["status"] == "degraded"


class TestRootEndpoint:
    """Test root endpoint functionality."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns service information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["service"] == "SD Multi-Modal Platform"
        assert data["version"] == "1.0.0-phase2"
        assert data["phase"] == "Phase 2: Backend Framework & Basic API Services"
        assert data["docs"] == f"{settings.API_PREFIX}/docs"
        assert data["health"] == f"{settings.API_PREFIX}/health"


class TestLoggingIntegration:
    """Test logging system integration."""

    def test_structured_logging_format(self, client, caplog):
        """Test that logs are properly structured."""
        # Clear any existing logs
        caplog.clear()

        # Make a request that will generate logs
        response = client.get(f"{settings.API_PREFIX}/health")

        assert response.status_code == 200

        # Check that logs were generated
        assert len(caplog.records) > 0

        # Find request-related log records
        request_logs = [
            record for record in caplog.records if hasattr(record, "request_id")
        ]

        assert len(request_logs) > 0

        # Check log record has expected attributes
        log_record = request_logs[0]
        assert hasattr(log_record, "request_id")
        assert len(log_record.request_id) == 8

    def test_request_logging_middleware(self, client, caplog):
        """Test request logging middleware captures requests."""
        caplog.clear()

        response = client.get(f"{settings.API_PREFIX}/health")
        assert response.status_code == 200

        # Should have both start and completion logs
        log_messages = [record.message for record in caplog.records]

        start_logs = [msg for msg in log_messages if "Request started" in msg]
        complete_logs = [msg for msg in log_messages if "Request completed" in msg]

        assert len(start_logs) >= 1
        assert len(complete_logs) >= 1


class TestPerformanceAndLoad:
    """Test performance characteristics."""

    def test_health_check_performance(self, client):
        """Test health check response time is reasonable."""
        start_time = time.time()

        response = client.get(f"{settings.API_PREFIX}/health")

        end_time = time.time()
        response_time = end_time - start_time

        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second

        # Check reported process time
        process_time_header = response.headers.get("X-Process-Time", "0s")
        reported_time = float(process_time_header[:-1])
        assert reported_time < 1.0

    def test_concurrent_requests(self, client):
        """Test handling multiple concurrent requests."""
        import concurrent.futures
        import threading

        def make_request():
            return client.get(f"{settings.API_PREFIX}/health")

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should have unique request IDs
        request_ids = [r.headers.get("X-Request-ID") for r in responses]
        assert len(set(request_ids)) == len(request_ids)  # All unique


# Benchmark tests
class TestBenchmarks:
    """Performance benchmark tests."""

    def test_health_check_benchmark(self, client):
        """Benchmark health check performance."""
        times = []

        for _ in range(50):
            start = time.time()
            response = client.get(f"{settings.API_PREFIX}/health")
            end = time.time()

            assert response.status_code == 200
            times.append(end - start)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)

        print(f"\nHealth Check Benchmark Results:")
        print(f"Average response time: {avg_time:.3f}s")
        print(f"Max response time: {max_time:.3f}s")
        print(f"Min response time: {min_time:.3f}s")
        print(f"95th percentile: {sorted(times)[int(0.95 * len(times))]:.3f}s")

        # Performance assertions
        assert avg_time < 0.1  # Average under 100ms
        assert max_time < 0.5  # Max under 500ms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
