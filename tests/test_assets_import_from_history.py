# tests/test_assets_import_from_history.py

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from PIL import Image

from services.history import get_history_store
from services.assets.asset_manager import AssetManager


@pytest.mark.integration
async def test_assets_import_from_history(client, mock_settings):
    # Reset singleton so it picks up patched settings.ASSETS_PATH for this test.
    AssetManager._instance = None
    AssetManager._initialized = False

    outputs_root = Path(str(mock_settings.OUTPUT_PATH))
    outputs_root.mkdir(parents=True, exist_ok=True)
    out_file = outputs_root / "history_out.png"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(out_file, format="PNG")

    store = get_history_store()
    history_id = f"task_{uuid.uuid4()}"
    store.record_completion(
        history_id=history_id,
        task_type="txt2img",
        run_mode="sync",
        user_id="user_a",
        input_params={"prompt": "test"},
        result_data={"result": {"images": [{"image_url": "/outputs/history_out.png"}]}},
    )

    resp = await client.post(
        "/api/v1/assets/import_from_history",
        json={"history_id": history_id, "category": "reference", "tags": ["from_history"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    imported = body["data"]["imported"]
    assert isinstance(imported, list) and len(imported) == 1
    assert imported[0].get("asset_id")

