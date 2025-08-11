# tests/conftest.py
"""
Pytest configuration and shared test fixtures.
Phase 2: Backend Framework & Basic API Services
"""

import logging
import os
import sys
import pytest
import asyncio
from pathlib import Path
from typing import Generator, AsyncGenerator, Optional
import time
from unittest.mock import patch, MagicMock
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set testing environment variables before importing app
os.environ.update(
    {
        "TESTING": "true",
        "LOG_LEVEL": "WARNING",
        "DEVICE": "cpu",
        "MAX_WORKERS": "1",
        "TORCH_DTYPE": "float32",
        "ENABLE_REQUEST_LOGGING": "true",
        "API_PREFIX": "/api/v1",
        "ALLOWED_ORIGINS": "http://localhost:3000,http://localhost:8080",
    }
)

# Import after setting environment
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    """
    Create a test client for FastAPI app with clean state for each test.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def mock_torch():
    """Mock PyTorch to avoid GPU dependencies in tests."""
    with patch("torch.cuda.is_available") as mock_cuda_available, patch(
        "torch.cuda.device_count"
    ) as mock_device_count, patch(
        "torch.cuda.current_device"
    ) as mock_current_device, patch(
        "torch.cuda.get_device_name"
    ) as mock_device_name, patch(
        "torch.cuda.get_device_properties"
    ) as mock_device_props:

        # Configure mocks
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_current_device.return_value = 0
        mock_device_name.return_value = "NVIDIA GeForce RTX 4090"

        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3  # 24GB
        mock_device_props.return_value = mock_props

        yield {
            "cuda_available": mock_cuda_available,
            "device_count": mock_device_count,
            "current_device": mock_current_device,
            "device_name": mock_device_name,
            "device_properties": mock_device_props,
        }


@pytest.fixture(scope="function")
def mock_torch_no_cuda():
    """Mock PyTorch with no CUDA support."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture(scope="function")
def temp_directories(tmp_path):
    """Create temporary directories for testing."""
    dirs = {
        "models": tmp_path / "models",
        "outputs": tmp_path / "outputs",
        "assets": tmp_path / "assets",
        "logs": tmp_path / "logs",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


@pytest.fixture(scope="function")
def mock_settings(temp_directories):
    """Mock settings with temporary directories."""
    with patch.object(
        settings, "OUTPUT_PATH", temp_directories["outputs"]
    ), patch.object(
        settings, "SD_MODEL_PATH", temp_directories["models"]
    ), patch.object(
        settings, "ASSETS_PATH", temp_directories["assets"]
    ):
        yield settings


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""

    # Clear all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Reset to default level
    root_logger.setLevel(logging.WARNING)

    yield

    # Cleanup after test
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


@pytest.fixture(scope="function")
def capture_logs():
    """Capture logs for testing logging functionality."""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    yield log_capture

    # Cleanup
    root_logger.removeHandler(handler)


# Test data fixtures
@pytest.fixture
def sample_health_response():
    """Sample health check response structure."""
    return {
        "status": "healthy",
        "timestamp": 1691234567.123,
        "service": {"name": "SD Multi-Modal Platform", "version": "1.0.0-phase2"},
        "system": {
            "platform": "Linux-5.4.0-test",
            "python_version": "3.10.0",
            "device": "cuda",
            "cuda_available": True,
        },
    }


@pytest.fixture
def sample_error_response():
    """Sample error response structure."""
    return {
        "success": False,
        "error": "Test error message",
        "status_code": 500,
        "request_id": "test123",
        "timestamp": 1691234567.123,
    }


# Utility functions for tests
def assert_response_structure(response_data: dict, required_fields: list):
    """Assert that response has required fields."""
    for field in required_fields:
        assert field in response_data, f"Required field '{field}' missing from response"


def assert_request_headers(response):
    """Assert that response has required request tracking headers."""
    assert "X-Request-ID" in response.headers, "X-Request-ID header missing"
    assert "X-Process-Time" in response.headers, "X-Process-Time header missing"

    # Validate header formats
    request_id = response.headers["X-Request-ID"]
    assert (
        len(request_id) == 8
    ), f"Request ID should be 8 characters, got {len(request_id)}"

    process_time = response.headers["X-Process-Time"]
    assert process_time.endswith("s"), "Process time should end with 's'"

    # Extract numeric value and validate
    time_value = float(process_time[:-1])
    assert time_value >= 0, "Process time should be non-negative"


def assert_error_response_format(response_data: dict):
    """Assert that error response follows standard format."""
    required_fields = ["success", "error", "request_id", "timestamp"]
    assert_response_structure(response_data, required_fields)
    assert response_data["success"] is False, "Error response should have success=False"


# Performance testing utilities
class PerformanceTimer:
    """Context manager for measuring performance."""

    start_time = Optional[float]
    end_time = Optional[float]
    duration = Optional[float]

    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.time()
        # Guard against None to satisfy the type checker and avoid runtime errors
        if self.start_time is None:
            raise RuntimeError("PerformanceTimer used without __enter__ being called.")
        self.duration = self.end_time - self.start_time  # type: ignore
        # return None / False means exceptions (if any) are not suppressed


@pytest.fixture
def performance_timer():
    """Fixture for measuring test performance."""
    return PerformanceTimer


# Skip markers for conditional tests
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    try:
        return pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        )
    except ImportError:
        return pytest.mark.skip(reason="PyTorch not installed")


def skip_if_no_gpu_memory():
    """Skip test if insufficient GPU memory."""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return pytest.mark.skipif(
                props.total_memory < 8 * 1024**3,  # 8GB
                reason="Insufficient GPU memory",
            )
        return pytest.mark.skip(reason="CUDA not available")
    except ImportError:
        return pytest.mark.skip(reason="PyTorch not installed")
