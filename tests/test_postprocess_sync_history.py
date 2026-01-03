# tests/test_postprocess_sync_history.py

from __future__ import annotations

import pytest

from utils.image_utils import create_test_image, pil_image_to_base64


@pytest.mark.integration
async def test_sync_upscale_and_face_restore_write_history(client, mock_settings):
    img = create_test_image(64, 64, "RGB")
    b64 = pil_image_to_base64(img)

    up = await client.post(
        "/api/v1/upscale/",
        json={"image": b64, "scale": 2, "model": "RealESRGAN_x4plus", "user_id": "user_a"},
    )
    assert up.status_code == 200
    up_body = up.json()
    assert up_body["success"] is True
    assert isinstance(up_body.get("data", {}).get("task_id"), str)

    fr = await client.post(
        "/api/v1/face_restore/",
        json={"image": b64, "model": "GFPGAN_v1.4", "upscale": 2, "user_id": "user_a"},
    )
    assert fr.status_code == 200
    fr_body = fr.json()
    assert fr_body["success"] is True
    assert isinstance(fr_body.get("data", {}).get("task_id"), str)

    hist_up = await client.get("/api/v1/history/list?task_type=upscale&user_id=user_a&limit=50")
    assert hist_up.status_code == 200
    ids = [r.get("task_id") for r in hist_up.json()["data"]["records"]]
    assert up_body["data"]["task_id"] in ids

    hist_fr = await client.get("/api/v1/history/list?task_type=face_restore&user_id=user_a&limit=50")
    assert hist_fr.status_code == 200
    ids = [r.get("task_id") for r in hist_fr.json()["data"]["records"]]
    assert fr_body["data"]["task_id"] in ids

