# tests/test_queue_asset_id_requests.py

import base64
import io
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.queue_requests import (
    QueueFaceRestoreRequest,
    QueueImg2ImgRequest,
    QueueInpaintRequest,
    QueueUpscaleRequest,
)


def _tiny_png_data_url() -> str:
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    b = io.BytesIO()
    img.save(b, format="PNG")
    data = b.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


@pytest.mark.unit
def test_queue_img2img_accepts_init_asset_id():
    asset_id = str(uuid.uuid4())
    req = QueueImg2ImgRequest(prompt="test", init_asset_id=asset_id)
    assert req.init_asset_id == asset_id


@pytest.mark.unit
def test_queue_img2img_rejects_multiple_init_sources():
    asset_id = str(uuid.uuid4())
    with pytest.raises(ValidationError):
        QueueImg2ImgRequest(prompt="test", init_asset_id=asset_id, init_image=_tiny_png_data_url())


@pytest.mark.unit
def test_queue_img2img_image_path_must_be_allowed(mock_settings, tmp_path):
    assets_root = Path(mock_settings.ASSETS_PATH)
    outputs_root = Path(str(mock_settings.OUTPUT_PATH))

    allowed = assets_root / "allowed.png"
    Image.new("RGB", (1, 1), color=(0, 255, 0)).save(allowed, format="PNG")
    req = QueueImg2ImgRequest(prompt="test", image_path=str(allowed))
    assert req.image_path == str(allowed.resolve())

    not_allowed = tmp_path / "not_allowed.png"
    Image.new("RGB", (1, 1), color=(0, 0, 255)).save(not_allowed, format="PNG")
    with pytest.raises(ValidationError):
        QueueImg2ImgRequest(prompt="test", image_path=str(not_allowed))

    # Smoke: output path is allowed too.
    out = outputs_root / "out.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color=(255, 255, 0)).save(out, format="PNG")
    req2 = QueueImg2ImgRequest(prompt="test", image_path=str(out))
    assert req2.image_path == str(out.resolve())


@pytest.mark.unit
def test_queue_img2img_control_asset_id_is_supported():
    init_asset_id = str(uuid.uuid4())
    control_asset_id = str(uuid.uuid4())
    req = QueueImg2ImgRequest(
        prompt="test",
        init_asset_id=init_asset_id,
        controlnet={"type": "canny", "preprocess": True, "strength": 1.0},
        control_asset_id=control_asset_id,
    )
    assert req.controlnet is not None
    assert req.controlnet.asset_id == control_asset_id


@pytest.mark.unit
def test_queue_inpaint_accepts_asset_ids():
    req = QueueInpaintRequest(
        prompt="test",
        init_asset_id=str(uuid.uuid4()),
        mask_asset_id=str(uuid.uuid4()),
    )
    assert req.init_asset_id
    assert req.mask_asset_id


@pytest.mark.unit
def test_queue_upscale_accepts_image_asset_id():
    asset_id = str(uuid.uuid4())
    req = QueueUpscaleRequest(image_asset_id=asset_id, scale=4, model="RealESRGAN_x4plus")
    assert req.image_asset_id == asset_id


@pytest.mark.unit
def test_queue_face_restore_accepts_image_asset_id():
    asset_id = str(uuid.uuid4())
    req = QueueFaceRestoreRequest(image_asset_id=asset_id, model="GFPGAN_v1.4", upscale=2)
    assert req.image_asset_id == asset_id


@pytest.mark.integration
async def test_queue_enqueue_accepts_img2img_asset_id(client):
    mock_manager = MagicMock()
    mock_manager.enqueue_task = AsyncMock(return_value="task_123")
    mock_manager.get_task_status = AsyncMock(return_value=None)
    mock_manager.task_store = MagicMock()
    mock_manager.task_store.update_task_status = AsyncMock(return_value=True)

    with patch("app.api.v1.queue.CELERY_AVAILABLE", True), patch(
        "app.api.v1.queue.celery_app"
    ) as mock_celery, patch(
        "app.api.v1.queue.get_queue_manager", new_callable=AsyncMock
    ) as mock_get_manager, patch(
        "app.api.v1.queue._get_queue_position", new_callable=AsyncMock
    ) as mock_position:
        mock_get_manager.return_value = mock_manager
        mock_position.return_value = None
        mock_celery.send_task.return_value = None

        asset_id = str(uuid.uuid4())
        resp = await client.post(
            "/api/v1/queue/enqueue",
            json={
                "task_type": "img2img",
                "parameters": {"prompt": "test", "init_asset_id": asset_id, "strength": 0.75},
                "priority": "normal",
                "user_id": "test_user",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("success") is True

        _args, kwargs = mock_manager.enqueue_task.call_args
        assert kwargs["task_type"] == "img2img"
        assert kwargs["input_params"]["init_asset_id"] == asset_id


@pytest.mark.integration
async def test_queue_enqueue_accepts_upscale_image_asset_id(client):
    mock_manager = MagicMock()
    mock_manager.enqueue_task = AsyncMock(return_value="task_123")
    mock_manager.get_task_status = AsyncMock(return_value=None)
    mock_manager.task_store = MagicMock()
    mock_manager.task_store.update_task_status = AsyncMock(return_value=True)

    with patch("app.api.v1.queue.CELERY_AVAILABLE", True), patch(
        "app.api.v1.queue.celery_app"
    ) as mock_celery, patch(
        "app.api.v1.queue.get_queue_manager", new_callable=AsyncMock
    ) as mock_get_manager, patch(
        "app.api.v1.queue._get_queue_position", new_callable=AsyncMock
    ) as mock_position:
        mock_get_manager.return_value = mock_manager
        mock_position.return_value = None
        mock_celery.send_task.return_value = None

        asset_id = str(uuid.uuid4())
        resp = await client.post(
            "/api/v1/queue/enqueue",
            json={
                "task_type": "upscale",
                "parameters": {"image_asset_id": asset_id, "scale": 4, "model": "RealESRGAN_x4plus"},
                "priority": "normal",
                "user_id": "test_user",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("success") is True

        _args, kwargs = mock_manager.enqueue_task.call_args
        assert kwargs["task_type"] == "upscale"
        assert kwargs["input_params"]["image_asset_id"] == asset_id


@pytest.mark.integration
async def test_queue_enqueue_accepts_face_restore_image_asset_id(client):
    mock_manager = MagicMock()
    mock_manager.enqueue_task = AsyncMock(return_value="task_123")
    mock_manager.get_task_status = AsyncMock(return_value=None)
    mock_manager.task_store = MagicMock()
    mock_manager.task_store.update_task_status = AsyncMock(return_value=True)

    with patch("app.api.v1.queue.CELERY_AVAILABLE", True), patch(
        "app.api.v1.queue.celery_app"
    ) as mock_celery, patch(
        "app.api.v1.queue.get_queue_manager", new_callable=AsyncMock
    ) as mock_get_manager, patch(
        "app.api.v1.queue._get_queue_position", new_callable=AsyncMock
    ) as mock_position:
        mock_get_manager.return_value = mock_manager
        mock_position.return_value = None
        mock_celery.send_task.return_value = None

        asset_id = str(uuid.uuid4())
        resp = await client.post(
            "/api/v1/queue/enqueue",
            json={
                "task_type": "face_restore",
                "parameters": {"image_asset_id": asset_id, "model": "GFPGAN_v1.4", "upscale": 2},
                "priority": "normal",
                "user_id": "test_user",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("success") is True

        _args, kwargs = mock_manager.enqueue_task.call_args
        assert kwargs["task_type"] == "face_restore"
        assert kwargs["input_params"]["image_asset_id"] == asset_id
