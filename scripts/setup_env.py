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


def setup_environment():
    """Setup development environment"""
    print("🚀 Setting up Multi-Modal Lab environment...")

    # Check if .env exists
    env_file = Path(".env")
    load_dotenv(env_file)

    if not env_file.exists():
        print(
            "❌ .env file not found. Please copy .env.example to .env and update AI_CACHE_ROOT"
        )
        sys.exit(1)

    ai_cache_root = Path(shared_cache.cache_root)
    # Create necessary directories
    directories = [
        ai_cache_root / "upload",
        ai_cache_root / "outputs",
        ai_cache_root / "cache",
        ai_cache_root / "vector_db",
        ai_cache_root / "logs",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

    print("✅ Environment setup completed!")
    print("\nNext steps:")
    print("1. Update AI_CACHE_ROOT in .env to point to your ai_warehouse")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: uvicorn backend.app.main:app --reload")


if __name__ == "__main__":
    setup_environment()
