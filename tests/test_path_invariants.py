# tests/test_path_invariants.py
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import settings
from app.shared_cache import shared_cache


@pytest.mark.unit
def test_storage_paths_follow_data_model_structure_spec():
    expected_model_root = Path("/mnt/c/ai_models")
    expected_cache_root = Path("/mnt/c/ai_cache")
    expected_output_root = Path("/mnt/data/training/runs/sd-multimodal-platform")

    assert Path(settings.AI_MODELS_ROOT) == expected_model_root
    assert Path(settings.AI_CACHE_ROOT) == expected_cache_root
    assert Path(settings.AI_OUTPUT_ROOT) == expected_output_root

    assert Path(settings.MODELS_PATH) == expected_model_root
    assert Path(settings.CONTROLNET_PATH) == expected_model_root / "controlnet"
    assert Path(settings.UPSCALE_MODELS_PATH) == expected_model_root / "upscale"
    assert Path(settings.FACE_RESTORE_MODELS_PATH) == expected_model_root / "face-restore"

    assert Path(settings.OUTPUT_PATH) == expected_output_root / "outputs"
    assert Path(settings.ASSETS_PATH) == expected_output_root / "assets"
    assert Path(settings.LOG_FILE_PATH) == expected_output_root / "logs" / "app.log"


@pytest.mark.unit
def test_cache_env_vars_are_derived_from_ai_cache_root():
    assert shared_cache.cache_root == "/mnt/c/ai_cache"
    assert shared_cache.models_root == "/mnt/c/ai_models"

    cache_info = shared_cache.get_cache_info()
    for key, expected in cache_info.items():
        assert os.environ.get(key) == expected

