# tests/test_cooperative_cancellation.py

from __future__ import annotations

import pytest
import numpy as np
from PIL import Image


@pytest.mark.unit
def test_realesrgan_wrapper_abort_check_breaks_tile_loop():
    from services.postprocess.upscale_service import RealESRGANWrapper

    class DummyUpsampler:
        def enhance(self, img, outscale=None):  # type: ignore[no-untyped-def]
            return img, None

    wrapper = RealESRGANWrapper(model_name="dummy", model_path="/dev/null", scale=1)
    wrapper.upsampler = DummyUpsampler()
    wrapper.tile = 64

    img = np.zeros((128, 128, 3), dtype=np.uint8)

    calls = {"n": 0}

    def abort_check() -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        wrapper._tile_enhance(img, outscale=1, abort_check=abort_check)


@pytest.mark.unit
async def test_upscale_service_abort_check_stops_immediately():
    from services.postprocess.upscale_service import UpscaleService

    service = UpscaleService()

    def abort_check() -> None:
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        await service.upscale_image(image=Image.new("RGB", (8, 8), color=(1, 2, 3)), abort_check=abort_check)


@pytest.mark.unit
async def test_face_restore_service_abort_check_stops_immediately():
    from services.postprocess.face_restore_service import FaceRestoreService

    service = FaceRestoreService()

    def abort_check() -> None:
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        await service.restore_faces(image=Image.new("RGB", (8, 8), color=(1, 2, 3)), abort_check=abort_check)


@pytest.mark.unit
@pytest.mark.parametrize(
    "method_name,pipeline_attr,kwargs",
    [
        (
            "generate_image",
            "current_pipeline",
            {"prompt": "a", "width": 64, "height": 64},
        ),
        (
            "generate_img2img",
            "current_img2img_pipeline",
            {"prompt": "a", "image": Image.new("RGB", (8, 8), color=(1, 2, 3))},
        ),
        (
            "generate_inpaint",
            "current_inpaint_pipeline",
            {
                "prompt": "a",
                "image": Image.new("RGB", (8, 8), color=(1, 2, 3)),
                "mask_image": Image.new("L", (8, 8), color=0),
            },
        ),
    ],
)
async def test_sd_models_progress_callback_exception_propagates(
    method_name: str, pipeline_attr: str, kwargs: dict
):
    from services.models.sd_models import get_model_manager

    class TaskCancelledError(RuntimeError):
        pass

    class DummyPipeline:
        def __call__(self, **pipeline_kwargs):  # type: ignore[no-untyped-def]
            callback = pipeline_kwargs.get("callback")
            if callable(callback):
                callback(0, 0, None)
            return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()

    manager = get_model_manager()
    prior_pipeline = getattr(manager, pipeline_attr, None)
    prior_model = getattr(manager, "current_model", None)
    try:
        setattr(manager, pipeline_attr, DummyPipeline())
        manager.current_model = "sd-1.5"

        def progress_callback(step: int, total_steps: int) -> None:
            raise TaskCancelledError("cancelled")

        method = getattr(manager, method_name)
        with pytest.raises(TaskCancelledError, match="cancelled"):
            await method(
                **kwargs,
                num_inference_steps=3,
                callback_steps=1,
                progress_callback=progress_callback,
            )
    finally:
        setattr(manager, pipeline_attr, prior_pipeline)
        manager.current_model = prior_model


@pytest.mark.unit
async def test_controlnet_progress_callback_exception_propagates():
    from services.processors.controlnet_service import (
        ControlNetProcessor,
        get_controlnet_manager,
    )

    class TaskCancelledError(RuntimeError):
        pass

    class DummyPipeline:
        def __call__(self, **pipeline_kwargs):  # type: ignore[no-untyped-def]
            callback = pipeline_kwargs.get("callback")
            if callable(callback):
                callback(0, 0, None)
            return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()

    manager = get_controlnet_manager()
    prior_pipeline = manager.current_pipeline
    prior_key = manager.current_pipeline_key
    prior_processors = manager.processors
    try:
        manager.current_pipeline = DummyPipeline()
        manager.current_pipeline_key = ("base_model", "canny", "img2img")
        manager.processors = {"sd:canny": ControlNetProcessor("canny", "/dev/null")}

        def progress_callback(step: int, total_steps: int) -> None:
            raise TaskCancelledError("cancelled")

        with pytest.raises(TaskCancelledError, match="cancelled"):
            await manager.generate_with_controlnet(
                prompt="a",
                init_image=Image.new("RGB", (8, 8), color=(1, 2, 3)),
                control_image=Image.new("RGB", (8, 8), color=(4, 5, 6)),
                controlnet_type="canny",
                num_inference_steps=3,
                controlnet_params={"preprocess": False},
                progress_callback=progress_callback,
            )
    finally:
        manager.current_pipeline = prior_pipeline
        manager.current_pipeline_key = prior_key
        manager.processors = prior_processors
