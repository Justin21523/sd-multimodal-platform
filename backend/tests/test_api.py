# backend/tests/test_api.py
"""
Backend API Test Suite

Comprehensive tests for API endpoints including:
- Health check and status endpoints
- Text-to-image generation
- Model management operations
- Error handling scenarios
"""

import pytest
import asyncio
import json
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.main import app
from backend.config.settings import settings
from backend.core.sd_pipeline import sd_manager

# Test client
client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints"""

    def test_root_endpoint(self):
        """Test API root endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "SD Multimodal Platform API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "device" in data
        assert "model_loaded" in data


class TestModelManagement:
    """Test model management endpoints"""

    def test_get_model_info(self):
        """Test model information endpoint"""
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "model_info" in data
        assert "available_models" in data

    def test_get_model_status(self):
        """Test detailed model status"""
        response = client.get("/api/v1/models/status")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "model_status" in data
        assert "memory_info" in data
        assert "system_info" in data

    def test_get_generation_parameters(self):
        """Test generation parameters endpoint"""
        response = client.get("/api/v1/txt2img/parameters")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "parameters" in data
        assert "schedulers" in data

        # Check parameter ranges
        params = data["parameters"]
        assert "width" in params
        assert "height" in params
        assert params["width"]["min"] == 64
        assert params["width"]["max"] == 2048


class TestTextToImage:
    """Test text-to-image generation endpoints"""

    @patch("backend.core.sd_pipeline.sd_manager.generate_image")
    def test_text_to_image_generation_mock(self, mock_generate):
        """Test text-to-image generation with mocked pipeline"""
        from PIL import Image

        # Mock successful generation
        mock_image = Image.new("RGB", (512, 512), color="red")
        mock_generate.return_value = [mock_image]

        request_data = {
            "prompt": "A beautiful sunset over mountains",
            "negative_prompt": "blurry, low quality",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "batch_size": 1,
        }

        response = client.post("/api/v1/txt2img/generate", json=request_data)

        if response.status_code != 200:
            print(f"Response: {response.text}")

        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert len(data["images"]) == 1
        assert "generation_time" in data
        assert "parameters" in data

    def test_text_to_image_validation(self):
        """Test input validation for text-to-image"""
        # Test missing prompt
        response = client.post("/api/v1/txt2img/generate", json={})
        assert response.status_code == 422  # Validation error

        # Test invalid dimensions
        request_data = {
            "prompt": "test",
            "width": 50,  # Too small
            "height": 3000,  # Too large
        }

        response = client.post("/api/v1/txt2img/generate", json=request_data)
        assert response.status_code == 422

    def test_batch_generation_validation(self):
        """Test batch size validation"""
        request_data = {
            "prompt": "test prompt",
            "batch_size": settings.max_batch_size + 1,  # Exceeds limit
        }

        response = client.post("/api/v1/txt2img/generate", json=request_data)
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_invalid_model_switch(self):
        """Test switching to non-existent model"""
        request_data = {"model_id": "non-existent-model"}

        response = client.post("/api/v1/models/switch", json=request_data)
        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert "error" in data

    def test_invalid_scheduler_change(self):
        """Test changing to invalid scheduler"""
        request_data = {"scheduler": "invalid-scheduler"}

        response = client.post("/api/v1/models/scheduler", json=request_data)
        assert response.status_code == 422  # Validation error
