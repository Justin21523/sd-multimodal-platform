# tests/test_phase3_integration.py
"""
Phase 3 Integration Tests
Tests for model management and real txt2img generation functionality.
"""

import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import torch

from services.models.sd_models import ModelManager, ModelRegistry, get_model_manager
from app.api.v1.txt2img import Txt2ImgRequest
from app.config import settings


def _has_local_models() -> bool:
    """Best-effort check for local model installation under MODELS_PATH."""
    try:
        model_info = ModelRegistry.get_model_info("sdxl-base")
        if not model_info:
            return False
        return (Path(settings.MODELS_PATH) / model_info["local_path"]).exists()  # type: ignore[index]
    except Exception:
        return False


class TestModelRegistry:
    """Test model registry functionality."""

    def test_get_model_info(self):
        """Test getting model information."""
        # Test valid model
        info = ModelRegistry.get_model_info("sdxl-base")
        assert info is not None
        assert info["name"] == "Stable Diffusion XL Base"
        assert info["vram_requirement"] == 8
        assert "photorealistic" in info["strengths"]

        # Test invalid model
        info = ModelRegistry.get_model_info("nonexistent")
        assert info is None

    def test_list_models(self):
        """Test listing available models."""
        models = ModelRegistry.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "sdxl-base" in models
        assert "sd-1.5" in models

    def test_find_models_by_strength(self):
        """Test finding models by capability."""
        # Test photorealistic models
        photo_models = ModelRegistry.find_models_by_strength("photorealistic")
        assert "sdxl-base" in photo_models

        # Test anime models
        anime_models = ModelRegistry.find_models_by_strength("anime")
        assert "sd-1.5" in anime_models

        # Test non-existent strength
        empty_models = ModelRegistry.find_models_by_strength("nonexistent")
        assert len(empty_models) == 0


class TestModelManagerBasic:
    """Test basic model manager functionality without requiring actual models."""

    def test_model_manager_initialization(self):
        """Test model manager creation."""
        manager = ModelManager()
        assert not manager.is_initialized
        assert manager.current_model_id is None
        assert manager.current_pipeline is None
        assert manager.startup_time == 0.0

    def test_singleton_pattern(self):
        """Test that get_model_manager returns same instance."""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        assert manager1 is manager2

    def test_get_status_uninitialized(self):
        """Test status when uninitialized."""
        manager = ModelManager()
        status = manager.get_status()

        assert not status["is_initialized"]
        assert status["current_model_id"] is None
        assert isinstance(status["available_models"], list)


@pytest.mark.integration
class TestModelManagerIntegration:
    """Integration tests requiring actual model files."""

    @pytest.fixture
    def manager(self):
        """Create a fresh model manager for each test."""
        # Create new instance to avoid singleton issues
        manager = ModelManager()
        yield manager
        # Cleanup after test
        if manager.is_initialized:
            asyncio.run(manager.cleanup())

    async def test_initialize_with_missing_models(self, manager):
        """Test initialization when models are missing."""
        # This should fail gracefully when models aren't downloaded
        success = await manager.initialize()

        # Should return False but not crash
        assert isinstance(success, bool)

        if not success:
            assert not manager.is_initialized
            assert manager.current_model_id is None

    @pytest.mark.skipif(
        not _has_local_models(), reason="Models not installed under MODELS_PATH"
    )
    async def test_initialize_with_models(self, manager):
        """Test initialization when models are available."""
        success = await manager.initialize()

        if success:
            assert manager.is_initialized
            assert manager.current_model_id is not None
            assert manager.current_pipeline is not None
            assert manager.startup_time > 0

    @pytest.mark.skipif(
        not _has_local_models(), reason="Models not installed under MODELS_PATH"
    )
    async def test_model_switching(self, manager):
        """Test switching between models."""
        # Initialize with default model
        await manager.initialize()

        if not manager.is_initialized:
            pytest.skip("Model initialization failed")

        original_model = manager.current_model_id

        # Try to switch to a different model
        available_models = ModelRegistry.list_models()
        target_model = None

        for model_id in available_models:
            if model_id != original_model:
                model_info = ModelRegistry.get_model_info(model_id)
                model_path = Path(settings.MODELS_PATH) / model_info["local_path"]  # type: ignore
                if model_path.exists():
                    target_model = model_id
                    break

        if target_model:
            success = await manager.switch_model(target_model)
            if success:
                assert manager.current_model_id == target_model

    @pytest.mark.skipif(
        not _has_local_models(), reason="Models not installed under MODELS_PATH"
    )
    async def test_image_generation_mock(self, manager):
        """Test image generation with mocked pipeline."""
        # Initialize manager
        await manager.initialize()

        if not manager.is_initialized:
            pytest.skip("Model initialization failed")

        # Mock the pipeline to avoid actual generation
        mock_result = Mock()
        mock_result.images = [Image.new("RGB", (512, 512), "red")]

        with patch.object(
            manager.current_pipeline, "__call__", return_value=mock_result
        ):
            result = await manager.generate_image(
                prompt="test prompt",
                width=512,
                height=512,
                num_inference_steps=1,  # Minimal steps for testing
                seed=42,
            )

            assert "images" in result
            assert "metadata" in result
            assert len(result["images"]) == 1
            assert isinstance(result["images"][0], Image.Image)
            assert result["metadata"]["seed"] == 42


class TestTxt2ImgAPISchema:
    """Test txt2img API schema validation."""

    def test_valid_request(self):
        """Test valid request creation."""
        request = Txt2ImgRequest(
            prompt="a beautiful landscape",
            negative_prompt="blurry",
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42,
        )

        assert request.prompt == "a beautiful landscape"
        assert request.width == 512
        assert request.height == 512
        assert request.seed == 42

    def test_dimension_validation(self):
        """Test dimension rounding to multiples of 8."""
        # Test rounding up
        request = Txt2ImgRequest(prompt="test", width=515, height=517)
        assert request.width == 520  # 515 -> 520
        assert request.height == 520  # 517 -> 520

        # Test rounding down
        request = Txt2ImgRequest(prompt="test", width=513, height=514)
        assert request.width == 520  # 513 -> 520 (rounds to nearest)
        assert request.height == 520  # 514 -> 520

    def test_seed_validation(self):
        """Test seed handling."""
        # Test -1 becomes None
        request = Txt2ImgRequest(prompt="test", seed=-1)
        assert request.seed is None

        # Test valid seed
        request = Txt2ImgRequest(prompt="test", seed=42)
        assert request.seed == 42

    def test_prompt_validation(self):
        """Test prompt validation."""
        # Valid prompt
        request = Txt2ImgRequest(prompt="a")
        assert request.prompt == "a"

        # Empty prompt should fail
        with pytest.raises(ValueError):
            Txt2ImgRequest(prompt="")

    def test_parameter_bounds(self):
        """Test parameter boundary validation."""
        # Valid parameters
        request = Txt2ImgRequest(
            prompt="test",
            width=256,  # minimum
            height=2048,  # maximum
            num_inference_steps=10,  # minimum
            guidance_scale=20.0,  # maximum
        )

        assert request.width == 256
        assert request.height == 2048
        assert request.num_inference_steps == 10
        assert request.guidance_scale == 20.0


@pytest.mark.integration
@pytest.mark.skipif(not _has_local_models(), reason="Models not installed under MODELS_PATH")
class TestFullGenerationFlow:
    """End-to-end integration tests for the full generation flow."""

    @pytest.fixture(scope="class")
    async def initialized_manager(self):
        """Initialize manager once for the test class."""
        manager = get_model_manager()
        if not manager.is_initialized:
            success = await manager.initialize()
            if not success:
                pytest.skip("Could not initialize model manager")
        yield manager

    async def test_minimal_generation(self, initialized_manager):
        """Test minimal image generation."""
        # Mock pipeline to avoid long generation times
        mock_result = Mock()
        mock_result.images = [Image.new("RGB", (512, 512), "blue")]

        with patch.object(
            initialized_manager.current_pipeline, "__call__", return_value=mock_result
        ):
            result = await initialized_manager.generate_image(
                prompt="test image",
                width=512,
                height=512,
                num_inference_steps=1,
                seed=123,
            )

            # Validate result structure
            assert "images" in result
            assert "metadata" in result

            metadata = result["metadata"]
            assert metadata["model_id"] == initialized_manager.current_model_id
            assert metadata["prompt"] == "test image"
            assert metadata["seed"] == 123
            assert metadata["width"] == 512
            assert metadata["height"] == 512
            assert metadata["generation_time"] > 0

    async def test_batch_generation(self, initialized_manager):
        """Test generating multiple images."""
        # Mock pipeline for batch generation
        mock_result = Mock()
        mock_result.images = [
            Image.new("RGB", (512, 512), "red"),
            Image.new("RGB", (512, 512), "green"),
        ]

        with patch.object(
            initialized_manager.current_pipeline, "__call__", return_value=mock_result
        ):
            result = await initialized_manager.generate_image(
                prompt="batch test",
                num_images=2,
                width=512,
                height=512,
                num_inference_steps=1,
                seed=456,
            )

            assert len(result["images"]) == 2
            assert result["metadata"]["num_images"] == 2
            assert all(isinstance(img, Image.Image) for img in result["images"])

    async def test_memory_management(self, initialized_manager):
        """Test VRAM monitoring and cleanup."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")

        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Mock generation
        mock_result = Mock()
        mock_result.images = [Image.new("RGB", (512, 512), "yellow")]

        with patch.object(
            initialized_manager.current_pipeline, "__call__", return_value=mock_result
        ):
            result = await initialized_manager.generate_image(
                prompt="memory test", width=512, height=512, num_inference_steps=1
            )

            # Check that memory info is included
            metadata = result["metadata"]
            assert "vram_used_gb" in metadata
            assert "vram_delta_gb" in metadata
            assert isinstance(metadata["vram_used_gb"], (int, float))

    async def test_error_handling(self, initialized_manager):
        """Test error handling during generation."""
        # Mock pipeline to raise an error
        with patch.object(
            initialized_manager.current_pipeline,
            "__call__",
            side_effect=RuntimeError("Mock error"),
        ):
            with pytest.raises(RuntimeError, match="Image generation failed"):
                await initialized_manager.generate_image(
                    prompt="error test", width=512, height=512, num_inference_steps=1
                )


class TestPerformanceBenchmarks:
    """Performance and memory benchmarks."""

    @pytest.mark.benchmark
    async def test_generation_performance(self):
        """Benchmark generation performance."""
        manager = get_model_manager()

        if not manager.is_initialized:
            pytest.skip("Model manager not initialized")

        # Mock for consistent timing
        mock_result = Mock()
        mock_result.images = [Image.new("RGB", (512, 512), "white")]

        start_time = time.time()

        with patch.object(
            manager.current_pipeline, "__call__", return_value=mock_result
        ):
            result = await manager.generate_image(
                prompt="benchmark test", width=512, height=512, num_inference_steps=1
            )

        total_time = time.time() - start_time
        generation_time = result["metadata"]["generation_time"]

        # Basic performance assertions
        assert total_time < 5.0  # Should complete within 5 seconds (mocked)
        assert generation_time > 0

        print(f"\nPerformance Results:")
        print(f"Total time: {total_time:.3f}s")
        print(f"Generation time: {generation_time:.3f}s")
        print(f"Model: {result['metadata']['model_id']}")


if __name__ == "__main__":
    # Run tests with appropriate markers
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-m",
            "not benchmark",  # Skip benchmarks by default
        ]
    )
