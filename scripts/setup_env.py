#!/usr/bin/env python3
"""
Environment setup script for Multi-Modal Lab
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from app.shared_cache import shared_cache
from app.config import settings


def setup_environment():
    """Setup development environment"""
    print("🚀 Setting up sd-multimodal-platform environment...")

    # Check if .env exists
    env_file = Path(".env")
    load_dotenv(env_file)

    if not env_file.exists():
        print(
            "❌ .env file not found. Please copy .env.example to .env and update AI_CACHE_ROOT"
        )
        sys.exit(1)

    # Create necessary directories (must follow ~/Desktop/data_model_structure.md)
    ai_cache_root = Path(shared_cache.cache_root)
    ai_models_root = Path(shared_cache.models_root)
    outputs_root = Path(settings.OUTPUT_PATH)
    assets_root = Path(settings.ASSETS_PATH)
    logs_root = Path(settings.LOG_FILE_PATH).parent

    directories = [
        ai_cache_root,
        ai_models_root,
        outputs_root,
        assets_root,
        logs_root,
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

    print("✅ Environment setup completed!")
    print("\nNext steps:")
    print("1. Copy .env.example → .env and adjust paths if needed")
    print("2. Activate conda env: conda activate ai_env")
    print("3. Install deps: pip install -r requirements.txt")
    print("4. Run API: uvicorn app.main:app --reload")


if __name__ == "__main__":
    setup_environment()
