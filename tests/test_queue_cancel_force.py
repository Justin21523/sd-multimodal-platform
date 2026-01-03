# tests/test_queue_cancel_force.py

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.integration
async def test_queue_cancel_soft_by_default(client):
    mock_manager = MagicMock()
    mock_manager.cancel_task = AsyncMock(return_value=True)

    with patch("app.api.v1.queue.celery_app") as mock_celery, patch(
        "app.api.v1.queue.get_queue_manager", new_callable=AsyncMock
    ) as mock_get_manager:
        mock_get_manager.return_value = mock_manager
        mock_celery.control.revoke = MagicMock(return_value=None)

        resp = await client.post("/api/v1/queue/cancel/task_123?user_id=test_user")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("success") is True

        mock_celery.control.revoke.assert_called_with("task_123", terminate=False)


@pytest.mark.integration
async def test_queue_cancel_force_terminates_when_requested(client):
    mock_manager = MagicMock()
    mock_manager.cancel_task = AsyncMock(return_value=True)

    with patch("app.api.v1.queue.celery_app") as mock_celery, patch(
        "app.api.v1.queue.get_queue_manager", new_callable=AsyncMock
    ) as mock_get_manager:
        mock_get_manager.return_value = mock_manager
        mock_celery.control.revoke = MagicMock(return_value=None)

        resp = await client.post(
            "/api/v1/queue/cancel/task_123?user_id=test_user&force=true"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("success") is True

        mock_celery.control.revoke.assert_called_with("task_123", terminate=True)

