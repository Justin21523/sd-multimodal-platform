# tests/test_history_api.py

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.history import get_history_store


@pytest.mark.integration
async def test_history_list_empty(client, mock_settings):
    resp = await client.get("/api/v1/history/list")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["records"] == []


@pytest.mark.integration
async def test_history_get_404(client, mock_settings):
    resp = await client.get("/api/v1/history/does_not_exist")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_history_write_and_get_and_export(client, mock_settings):
    store = get_history_store()
    history_id = f"task_{uuid.uuid4()}"
    store.record_completion(
        history_id=history_id,
        task_type="txt2img",
        run_mode="async",
        user_id="test_user",
        input_params={"prompt": "hello", "negative_prompt": "", "width": 512, "height": 512, "steps": 20},
        result_data={"result": {"images": [{"image_url": "/outputs/x.png", "image_path": "/tmp/x.png"}]}},
    )

    resp = await client.get(f"/api/v1/history/{history_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["history_id"] == history_id
    assert body["data"]["task_type"] == "txt2img"

    resp2 = await client.get(f"/api/v1/history/{history_id}/export")
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["success"] is True
    assert body2["data"]["history_id"] == history_id


@pytest.mark.integration
async def test_history_rerun_enqueues_task(client, mock_settings):
    store = get_history_store()
    history_id = f"task_{uuid.uuid4()}"
    store.record_completion(
        history_id=history_id,
        task_type="txt2img",
        run_mode="async",
        user_id="test_user",
        input_params={"prompt": "hello", "negative_prompt": "", "width": 512, "height": 512, "steps": 20},
        result_data={"result": {"images": [{"image_url": "/outputs/x.png"}]}},
    )

    mock_manager = MagicMock()
    mock_manager.enqueue_task = AsyncMock(return_value="rerun_task_123")

    with patch("app.api.v1.history.CELERY_AVAILABLE", True), patch(
        "app.api.v1.history.celery_app"
    ) as mock_celery, patch(
        "app.api.v1.history.get_queue_manager", new_callable=AsyncMock
    ) as mock_get_manager:
        mock_get_manager.return_value = mock_manager
        mock_celery.send_task.return_value = None

        resp = await client.post(
            f"/api/v1/history/{history_id}/rerun",
            json={"priority": "normal", "overrides": {"seed": 123}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["task_id"] == "rerun_task_123"

        _args, kwargs = mock_manager.enqueue_task.call_args
        assert kwargs["task_type"] == "txt2img"
        assert kwargs["input_params"]["seed"] == 123


@pytest.mark.integration
async def test_history_list_filters_and_cleanup(client, mock_settings):
    store = get_history_store()

    old_id = f"old_{uuid.uuid4()}"
    new_id = f"new_{uuid.uuid4()}"

    store.write_record(
        {
            "version": "history.v1",
            "history_id": old_id,
            "task_id": old_id,
            "task_type": "txt2img",
            "run_mode": "sync",
            "user_id": "user_a",
            "created_at": "2000-01-01T00:00:00+00:00",
            "input_params": {"prompt": "cat in a hat", "negative_prompt": "", "width": 512, "height": 512},
            "output_images": [{"image_url": "/outputs/old.png"}],
            "result_data": {"result": {"images": [{"image_url": "/outputs/old.png"}]}},
        }
    )
    store.write_record(
        {
            "version": "history.v1",
            "history_id": new_id,
            "task_id": new_id,
            "task_type": "img2img",
            "run_mode": "async",
            "user_id": "user_b",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_params": {"prompt": "dog", "negative_prompt": "", "strength": 0.75},
            "output_images": [{"image_url": "/outputs/new.png"}],
            "result_data": {"result": {"images": [{"image_url": "/outputs/new.png"}]}},
        }
    )

    resp = await client.get("/api/v1/history/list?q=cat")
    assert resp.status_code == 200
    body = resp.json()
    ids = [r.get("history_id") for r in body["data"]["records"]]
    assert old_id in ids
    assert new_id not in ids

    resp = await client.get("/api/v1/history/list?task_type=img2img")
    assert resp.status_code == 200
    body = resp.json()
    ids = [r.get("history_id") for r in body["data"]["records"]]
    assert new_id in ids
    assert old_id not in ids

    resp = await client.get("/api/v1/history/list?user_id=user_a")
    assert resp.status_code == 200
    body = resp.json()
    ids = [r.get("history_id") for r in body["data"]["records"]]
    assert old_id in ids
    assert new_id not in ids

    resp = await client.get("/api/v1/history/list?until=2001-01-01T00:00:00Z")
    assert resp.status_code == 200
    body = resp.json()
    ids = [r.get("history_id") for r in body["data"]["records"]]
    assert old_id in ids
    assert new_id not in ids

    resp = await client.delete("/api/v1/history/cleanup?older_than_days=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["deleted"] >= 1

    resp = await client.get(f"/api/v1/history/{old_id}")
    assert resp.status_code == 404

    resp = await client.get(f"/api/v1/history/{new_id}")
    assert resp.status_code == 200
