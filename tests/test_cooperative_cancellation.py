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

