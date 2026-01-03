#!/usr/bin/env python3
"""
Path consistency checks (highest priority).

Verifies all model/cache/output paths follow ~/Desktop/data_model_structure.md:
- models:  /mnt/c/ai_models
- caches:  /mnt/c/ai_cache
- outputs: /mnt/data/training/runs/sd-multimodal-platform/{outputs,assets,logs}

Also scans the repo for known anti-patterns that can accidentally write models
under the outputs tree (e.g. deriving `models` from `OUTPUT_PATH.parent`).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


sys.path.insert(0, str(_project_root()))

from app.config import settings  # noqa: E402
from app.shared_cache import shared_cache  # noqa: E402


EXPECTED_MODEL_ROOT = Path("/mnt/c/ai_models")
EXPECTED_CACHE_ROOT = Path("/mnt/c/ai_cache")
EXPECTED_OUTPUT_ROOT = Path("/mnt/data/training/runs/sd-multimodal-platform")


def _norm(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def _check_equal(errors: list[str], label: str, actual: str | Path, expected: Path):
    if _norm(actual) != _norm(expected):
        errors.append(f"{label}: expected '{expected}', got '{actual}'")


def _scan_for_antipatterns(repo_root: Path) -> list[str]:
    patterns = [
        ("derive-models-from-output-parent", re.compile(r"OUTPUT_PATH\)\.parent")),
        ("derive-models-from-output-parent-2", re.compile(r"Path\\(settings\\.OUTPUT_PATH\\)\\.parent")),
        ("output-path-models-combo", re.compile(r"OUTPUT_PATH.*\\bmodels\\b")),
        ("parent-slash-models", re.compile(r"parent\\s*/\\s*[\"']models[\"']")),
    ]

    python_roots = ["app", "services", "scripts", "utils", "tests"]
    errors: list[str] = []

    for root_name in python_roots:
        root = repo_root / root_name
        if not root.exists():
            continue

        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                # best-effort scan
                continue

            for name, pattern in patterns:
                if pattern.search(text):
                    rel = path.relative_to(repo_root)
                    errors.append(f"Anti-pattern '{name}' found in {rel}")

    return errors


def main() -> int:
    errors: list[str] = []

    # Roots (single source of truth)
    _check_equal(errors, "AI_MODELS_ROOT", settings.AI_MODELS_ROOT, EXPECTED_MODEL_ROOT)
    _check_equal(errors, "AI_CACHE_ROOT", settings.AI_CACHE_ROOT, EXPECTED_CACHE_ROOT)
    _check_equal(errors, "AI_OUTPUT_ROOT", settings.AI_OUTPUT_ROOT, EXPECTED_OUTPUT_ROOT)

    # Derived paths
    _check_equal(errors, "MODELS_PATH", settings.MODELS_PATH, EXPECTED_MODEL_ROOT)
    _check_equal(
        errors,
        "CONTROLNET_PATH",
        settings.CONTROLNET_PATH,
        EXPECTED_MODEL_ROOT / "controlnet",
    )
    _check_equal(
        errors,
        "UPSCALE_MODELS_PATH",
        settings.UPSCALE_MODELS_PATH,
        EXPECTED_MODEL_ROOT / "upscale",
    )
    _check_equal(
        errors,
        "FACE_RESTORE_MODELS_PATH",
        settings.FACE_RESTORE_MODELS_PATH,
        EXPECTED_MODEL_ROOT / "face-restore",
    )

    _check_equal(errors, "OUTPUT_PATH", settings.OUTPUT_PATH, EXPECTED_OUTPUT_ROOT / "outputs")
    _check_equal(errors, "ASSETS_PATH", settings.ASSETS_PATH, EXPECTED_OUTPUT_ROOT / "assets")
    _check_equal(
        errors,
        "LOG_FILE_PATH",
        settings.LOG_FILE_PATH,
        EXPECTED_OUTPUT_ROOT / "logs" / "app.log",
    )

    # Cache env vars (set by SharedCache)
    env_expected = shared_cache.get_cache_info()
    for key, expected in env_expected.items():
        actual = os.environ.get(key)
        if actual != expected:
            errors.append(f"env:{key}: expected '{expected}', got '{actual}'")

    errors.extend(_scan_for_antipatterns(_project_root()))

    if errors:
        print("❌ Path checks failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print("✅ Path checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

