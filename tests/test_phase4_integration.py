# tests/test_phase4_integration.py
"""
Comprehensive Phase 4 integration tests for ControlNet, img2img, and asset management
"""
import pytest
import asyncio
import base64
import tempfile
from pathlib import Path
from PIL import Image
import json
import time

from fastapi.testclient import TestClient
from app.main import app
from services.processors.controlnet_service import get_controlnet_manager
from services.assets.asset_manager import get_asset_manager
from utils.image_utils import pil_image_to_base64, create_test_image

client = TestClient(app)


class TestPhase4Integration:
    """Phase 4 comprehensive integration tests"""

    @pytest.fixture(autouse=True)
    async def setup_test_environment(self):
        """Setup test environment with mock assets"""
        # Initialize asset manager
        asset_manager = get_asset_manager()
        await asset_manager.initialize()

        # Create test images
        self.test_image = create_test_image(512, 512, "RGB")
        self.test_image_b64 = pil_image_to_base64(self.test_image)

        self.test_mask = create_test_image(512, 512, "L", fill_color=255)
        self.test_mask_b64 = pil_image_to_base64(self.test_mask)

        yield

        # Cleanup
        await asset_manager.cleanup_orphaned_assets()

    def test_health_check_includes_phase4_capabilities(self):
        """Test health check includes Phase 4 capabilities"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "capabilities" in data["data"]
        capabilities = data["data"]["capabilities"]

        # Phase 4 capabilities should be reported
        expected_capabilities = ["img2img", "inpaint", "controlnet", "asset_management"]
        for capability in expected_capabilities:
            assert capability in capabilities

    def test_img2img_api_structure(self):
        """Test img2img API endpoint structure"""
        # Test status endpoint
        response = client.get("/api/v1/img2img/status")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "img2img_available" in data["data"]
        assert "controlnet_status" in data["data"]

    def test_img2img_request_validation(self):
        """Test img2img request validation"""
        # Test valid request structure
        valid_request = {
            "prompt": "a beautiful landscape",
            "init_image": self.test_image_b64,
            "strength": 0.75,
        }

        # In minimal mode, should return appropriate error
        response = client.post("/api/v1/img2img/", json=valid_request)
        # Expect either success or 503 (service unavailable in minimal mode)
        assert response.status_code in [200, 503]

        # Test invalid request (missing init_image)
        invalid_request = {
            "prompt": "test prompt"
            # Missing init_image
        }

        response = client.post("/api/v1/img2img/", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_inpaint_request_validation(self):
        """Test inpaint request validation"""
        valid_request = {
            "prompt": "fix this area",
            "init_image": self.test_image_b64,
            "mask_image": self.test_mask_b64,
            "strength": 0.8,
        }

        response = client.post("/api/v1/img2img/inpaint", json=valid_request)
        assert response.status_code in [
            200,
            503,
        ]  # Success or unavailable in minimal mode

        # Test invalid mask
        invalid_request = {
            "prompt": "fix this area",
            "init_image": self.test_image_b64,
            "mask_image": "invalid_base64",
            "strength": 0.8,
        }

        response = client.post("/api/v1/img2img/inpaint", json=invalid_request)
        assert response.status_code == 422

    def test_controlnet_configuration_validation(self):
        """Test ControlNet configuration validation"""
        valid_controlnet_config = {
            "type": "canny",
            "image": self.test_image_b64,
            "strength": 1.0,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
        }

        request_with_controlnet = {
            "prompt": "a building with strong edges",
            "init_image": self.test_image_b64,
            "controlnet": valid_controlnet_config,
        }

        response = client.post("/api/v1/img2img/", json=request_with_controlnet)
        assert response.status_code in [200, 503]

        # Test invalid ControlNet type
        invalid_controlnet = {
            "type": "invalid_type",
            "image": self.test_image_b64,
            "strength": 1.0,
        }

        request_with_invalid_controlnet = {
            "prompt": "test",
            "init_image": self.test_image_b64,
            "controlnet": invalid_controlnet,
        }

        response = client.post("/api/v1/img2img/", json=request_with_invalid_controlnet)
        assert response.status_code == 422


class TestAssetManagement:
    """Asset management system tests"""

    @pytest.fixture(autouse=True)
    async def setup_asset_tests(self):
        """Setup asset management tests"""
        self.asset_manager = get_asset_manager()
        await self.asset_manager.initialize()

        # Create test asset
        self.test_image = create_test_image(256, 256, "RGB")
        self.test_image_bytes = self.image_to_bytes(self.test_image)

        yield

        # Cleanup test assets
        await self.asset_manager.cleanup_orphaned_assets()

    def image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL image to bytes"""
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()

    def test_asset_categories_endpoint(self):
        """Test asset categories API"""
        response = client.get("/api/v1/assets/categories")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "categories" in data["data"]
        assert "total_assets" in data["data"]

    def test_asset_upload_validation(self):
        """Test asset upload validation"""
        # Test file upload structure
        files = [("files", ("test.png", self.test_image_bytes, "image/png"))]

        response = client.post(
            "/api/v1/assets/upload",
            files=files,
            data={
                "category": "reference",
                "tags": "test,validation",
                "descriptions": "Test image for validation",
            },
        )

        # Should handle upload appropriately
        assert response.status_code in [200, 422]  # Success or validation error

    def test_asset_list_filtering(self):
        """Test asset listing with filters"""
        # Test basic listing
        response = client.get("/api/v1/assets/list")
        assert response.status_code == 200

        data = response.json()
        assert "assets" in data["data"]
        assert "pagination" in data["data"]

        # Test filtering by category
        response = client.get("/api/v1/assets/list?category=reference")
        assert response.status_code == 200

        # Test pagination
        response = client.get("/api/v1/assets/list?limit=10&offset=0")
        assert response.status_code == 200


class TestImageUtilities:
    """Test image processing utilities"""

    def test_base64_image_conversion(self):
        """Test base64 image conversion utilities"""
        from utils.image_utils import base64_to_pil_image, pil_image_to_base64

        # Create test image
        test_image = create_test_image(256, 256, "RGB")

        # Convert to base64 and back
        b64_string = pil_image_to_base64(test_image)
        recovered_image = base64_to_pil_image(b64_string)

        # Verify dimensions and mode
        assert recovered_image.size == test_image.size
        assert recovered_image.mode == test_image.mode

    def test_img2img_image_preparation(self):
        """Test img2img image preparation"""
        from utils.image_utils import prepare_img2img_image

        # Test with odd dimensions (should be rounded to multiple of 8)
        test_image = create_test_image(511, 511, "RGB")
        prepared = prepare_img2img_image(test_image, 512, 512)

        assert prepared.width % 8 == 0
        assert prepared.height % 8 == 0
        assert prepared.mode == "RGB"

    def test_inpaint_mask_preparation(self):
        """Test inpaint mask preparation"""
        from utils.image_utils import prepare_inpaint_mask

        init_image = create_test_image(512, 512, "RGB")
        mask_image = create_test_image(512, 512, "L", fill_color=128)

        prepared_init, prepared_mask = prepare_inpaint_mask(
            init_image, mask_image, blur_radius=2
        )

        assert prepared_init.size == prepared_mask.size
        assert prepared_init.mode == "RGB"
        assert prepared_mask.mode == "L"

    def test_controlnet_preprocessing(self):
        """Test ControlNet image preprocessing"""
        from utils.image_utils import preprocess_controlnet_image, create_canny_edges

        test_image = create_test_image(400, 600, "RGB")

        # Test preprocessing with aspect ratio maintenance
        processed = preprocess_controlnet_image(
            test_image, 512, 512, maintain_aspect=True
        )

        assert processed.size == (512, 512)
        assert processed.mode == "RGB"

        # Test Canny edge detection
        edges = create_canny_edges(test_image, 50, 200)
        assert edges.size == test_image.size
        assert edges.mode == "RGB"


class TestControlNetManager:
    """Test ControlNet manager functionality"""

    @pytest.fixture(autouse=True)
    async def setup_controlnet_tests(self):
        """Setup ControlNet tests"""
        self.controlnet_manager = get_controlnet_manager()
        yield
        # Cleanup
        await self.controlnet_manager.cleanup()

    def test_controlnet_manager_initialization(self):
        """Test ControlNet manager structure"""
        assert hasattr(self.controlnet_manager, "supported_types")
        assert hasattr(self.controlnet_manager, "processors")

        # Check supported types
        expected_types = ["canny", "openpose", "depth", "mlsd", "normal", "scribble"]
        for controlnet_type in expected_types:
            assert controlnet_type in self.controlnet_manager.supported_types

    def test_controlnet_status_reporting(self):
        """Test ControlNet status reporting"""
        status = self.controlnet_manager.get_status()

        assert "loaded_processors" in status
        assert "supported_types" in status
        assert "pipeline_loaded" in status
        assert "total_vram_usage" in status


class TestPerformanceBenchmarks:
    """Performance and memory benchmarks"""

    def test_api_response_times(self):
        """Test API response time benchmarks"""
        # Health check should be fast
        start_time = time.time()
        response = client.get("/api/v1/health")
        health_time = time.time() - start_time

        assert response.status_code == 200
        assert health_time < 0.1  # Health check should be under 100ms

        # Asset listing should be reasonable
        start_time = time.time()
        response = client.get("/api/v1/assets/list?limit=10")
        list_time = time.time() - start_time

        assert response.status_code == 200
        assert list_time < 1.0  # Asset listing should be under 1 second

    def test_image_processing_performance(self):
        """Test image processing performance"""
        from utils.image_utils import prepare_img2img_image, create_canny_edges

        # Test image preparation performance
        large_image = create_test_image(2048, 2048, "RGB")

        start_time = time.time()
        prepared = prepare_img2img_image(large_image, 1024, 1024)
        prep_time = time.time() - start_time

        assert prep_time < 2.0  # Should process 2048x2048 image in under 2 seconds
        assert prepared.size == (1024, 1024)

        # Test Canny edge detection performance
        start_time = time.time()
        edges = create_canny_edges(prepared)
        canny_time = time.time() - start_time

        assert canny_time < 1.0  # Canny detection should be under 1 second


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_image_data_handling(self):
        """Test handling of invalid image data"""
        invalid_requests = [
            {"prompt": "test", "init_image": "invalid_base64_data"},
            {"prompt": "test", "init_image": ""},  # Empty image data
            {
                "prompt": "",  # Empty prompt
                "init_image": "valid_but_will_fail_due_to_prompt",
            },
        ]

        for invalid_request in invalid_requests:
            response = client.post("/api/v1/img2img/", json=invalid_request)
            assert response.status_code == 422  # Validation error

    def test_large_file_upload_rejection(self):
        """Test rejection of files that are too large"""
        # Create a large dummy file (simulate > 10MB)
        large_file_data = b"x" * (11 * 1024 * 1024)  # 11MB

        files = [("files", ("large_file.png", large_file_data, "image/png"))]

        response = client.post(
            "/api/v1/assets/upload", files=files, data={"category": "reference"}
        )

        # Should either reject or handle gracefully
        assert response.status_code in [200, 413, 422]

        if response.status_code == 200:
            # If it returns 200, check that the large file was rejected
            data = response.json()
            assert len(data["data"]["failed_uploads"]) > 0


def create_test_image(
    width: int, height: int, mode: str = "RGB", fill_color=None
) -> Image.Image:
    """Create a test image for testing purposes"""
    if mode == "RGB":
        color = fill_color or (128, 128, 128)  # Gray
    elif mode == "L":
        color = fill_color or 128  # Gray
    elif mode == "RGBA":
        color = fill_color or (128, 128, 128, 255)  # Gray with alpha
    else:
        color = fill_color or 128

    return Image.new(mode, (width, height), color)


# Utility functions for testing
def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("🔥 Running Phase 4 Performance Benchmark...")

    # API Response Times
    print("\n📊 API Response Time Tests:")

    # Health check benchmark
    times = []
    for _ in range(10):
        start = time.time()
        response = client.get("/api/v1/health")
        times.append(time.time() - start)

    avg_health_time = sum(times) / len(times)
    print(f"  Health Check: {avg_health_time*1000:.1f}ms avg")

    # Asset listing benchmark
    times = []
    for _ in range(5):
        start = time.time()
        response = client.get("/api/v1/assets/list?limit=20")
        times.append(time.time() - start)

    avg_list_time = sum(times) / len(times)
    print(f"  Asset Listing: {avg_list_time*1000:.1f}ms avg")

    print("\n🧠 Image Processing Tests:")

    # Image processing benchmark
    test_image = create_test_image(1024, 1024, "RGB")

    start = time.time()
    from utils.image_utils import prepare_img2img_image

    prepared = prepare_img2img_image(test_image, 512, 512)
    prep_time = time.time() - start
    print(f"  Image Resize: {prep_time*1000:.1f}ms")

    start = time.time()
    from utils.image_utils import create_canny_edges

    edges = create_canny_edges(prepared)
    canny_time = time.time() - start
    print(f"  Canny Detection: {canny_time*1000:.1f}ms")

    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    # Run benchmark when script is executed directly
    run_performance_benchmark()
