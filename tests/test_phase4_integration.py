# tests/test_phase4_integration.py
"""
Lightweight Phase 4 integration checks (img2img + assets).

These tests avoid FastAPI TestClient threads and use the AsyncClient fixture.
They are written to pass in minimal mode (models not installed) by allowing 503s
on generation endpoints.
"""

from __future__ import annotations

import io

import pytest

from utils.image_utils import create_test_image, pil_image_to_base64
from services.assets.asset_manager import AssetManager


@pytest.mark.integration
class TestImg2ImgAndAssets:
    async def test_img2img_status(self, client):
        resp = await client.get("/api/v1/img2img/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "img2img_available" in data["data"]
        assert "controlnet_status" in data["data"]

    async def test_img2img_validation(self, client):
        test_image = create_test_image(256, 256, "RGB")
        b64 = pil_image_to_base64(test_image)

        # Missing init_image should fail validation
        resp = await client.post("/api/v1/img2img/", json={"prompt": "test"})
        assert resp.status_code == 422

        # Valid shape may succeed or be unavailable (models not installed)
        resp = await client.post(
            "/api/v1/img2img/",
            json={"prompt": "a beautiful landscape", "init_image": b64, "strength": 0.75},
        )
        assert resp.status_code in {200, 503}

    async def test_inpaint_validation(self, client):
        test_image = create_test_image(256, 256, "RGB")
        test_mask = create_test_image(256, 256, "L", fill_color=255)
        b64_img = pil_image_to_base64(test_image)
        b64_mask = pil_image_to_base64(test_mask)

        # Valid shape may succeed or be unavailable (models not installed)
        resp = await client.post(
            "/api/v1/img2img/inpaint",
            json={
                "prompt": "fix this area",
                "init_image": b64_img,
                "mask_image": b64_mask,
                "strength": 0.8,
            },
        )
        assert resp.status_code in {200, 503}

        # Invalid base64 should fail validation
        resp = await client.post(
            "/api/v1/img2img/inpaint",
            json={
                "prompt": "fix this area",
                "init_image": b64_img,
                "mask_image": "invalid_base64",
                "strength": 0.8,
            },
        )
        assert resp.status_code == 422

    async def test_img2img_accepts_init_asset_id(self, client, mock_settings):
        # Reset singleton so it picks up patched settings.ASSETS_PATH for this test.
        AssetManager._instance = None
        AssetManager._initialized = False
        try:
            test_image = create_test_image(256, 256, "RGB")
            buf = io.BytesIO()
            test_image.save(buf, format="PNG")

            upload = await client.post(
                "/api/v1/assets/upload",
                data={"category": "reference", "tags": "", "descriptions": ""},
                files=[("files", ("init.png", buf.getvalue(), "image/png"))],
            )
            assert upload.status_code == 200
            asset_id = upload.json()["data"]["uploaded_assets"][0]["asset_id"]

            resp = await client.post(
                "/api/v1/img2img/",
                json={
                    "prompt": "a beautiful landscape",
                    "init_asset_id": asset_id,
                    "strength": 0.75,
                },
            )
            assert resp.status_code in {200, 503}
        finally:
            AssetManager._instance = None
            AssetManager._initialized = False

    async def test_inpaint_accepts_asset_ids(self, client, mock_settings):
        # Reset singleton so it picks up patched settings.ASSETS_PATH for this test.
        AssetManager._instance = None
        AssetManager._initialized = False
        try:
            test_image = create_test_image(256, 256, "RGB")
            test_mask = create_test_image(256, 256, "L", fill_color=255)

            buf_img = io.BytesIO()
            test_image.save(buf_img, format="PNG")
            upload_img = await client.post(
                "/api/v1/assets/upload",
                data={"category": "reference", "tags": "", "descriptions": ""},
                files=[("files", ("init.png", buf_img.getvalue(), "image/png"))],
            )
            assert upload_img.status_code == 200
            init_asset_id = upload_img.json()["data"]["uploaded_assets"][0]["asset_id"]

            buf_mask = io.BytesIO()
            test_mask.save(buf_mask, format="PNG")
            upload_mask = await client.post(
                "/api/v1/assets/upload",
                data={"category": "mask", "tags": "", "descriptions": ""},
                files=[("files", ("mask.png", buf_mask.getvalue(), "image/png"))],
            )
            assert upload_mask.status_code == 200
            mask_asset_id = upload_mask.json()["data"]["uploaded_assets"][0]["asset_id"]

            resp = await client.post(
                "/api/v1/img2img/inpaint",
                json={
                    "prompt": "fix this area",
                    "init_asset_id": init_asset_id,
                    "mask_asset_id": mask_asset_id,
                    "strength": 0.8,
                },
            )
            assert resp.status_code in {200, 503}
        finally:
            AssetManager._instance = None
            AssetManager._initialized = False

    async def test_asset_categories_and_list(self, client, mock_settings):
        resp = await client.get("/api/v1/assets/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "categories" in data["data"]

        resp = await client.get("/api/v1/assets/list?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "assets" in data["data"]
        assert "pagination" in data["data"]
