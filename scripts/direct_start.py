#!/usr/bin/env python3
"""
Direct startup script that bypasses complex initialization.
Focuses on getting the API server running quickly.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def start_with_minimal_mode():
    """Start server in minimal mode."""
    print("🔧 Starting in MINIMAL MODE (no model loading)")
    print("This allows testing API structure without model initialization")
    print("-" * 50)

    # Set environment for minimal mode
    env = os.environ.copy()
    env.update({"MINIMAL_MODE": "true", "DEVICE": "cpu", "DEBUG": "true"})

    try:
        # Start uvicorn directly
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
            "--log-level",
            "info",
        ]

        print(f"Command: {' '.join(cmd)}")
        print("Starting server...")

        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        print(f"Server started with PID: {process.pid}")
        print("=" * 50)
        print("🌐 Server should be available at:")
        print("   Main: http://localhost:8000")
        print("   Health: http://localhost:8000/api/v1/health")
        print("   Docs: http://localhost:8000/api/v1/docs")
        print("=" * 50)
        print("Press Ctrl+C to stop")

        # Stream output
        try:
            for line in process.stdout:  # type: ignore
                print(line.rstrip())
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped")

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

    return True


def start_with_cpu_mode():
    """Start server with CPU-only mode."""
    print("💻 Starting in CPU MODE (with model loading)")
    print("This may take longer but includes full functionality")
    print("-" * 50)

    # Set environment for CPU mode
    env = os.environ.copy()
    env.update(
        {
            "DEVICE": "cpu",
            "ENABLE_CPU_OFFLOAD": "true",
            "USE_ATTENTION_SLICING": "true",
            "PRIMARY_MODEL": "sd-1.5",  # Use lighter model
            "DEBUG": "true",
        }
    )

    try:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--timeout-keep-alive",
            "120",
            "--log-level",
            "info",
        ]

        print(f"Command: {' '.join(cmd)}")
        print("Starting server (this may take 1-2 minutes)...")

        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        print(f"Server started with PID: {process.pid}")
        print("⏳ Waiting for model initialization...")

        # Stream output
        try:
            for line in process.stdout:  # type: ignore
                print(line.rstrip())
                # Look for success indicators
                if "Application startup completed" in line:
                    print("\n🎉 Server fully initialized!")
                elif "Model manager initialized" in line:
                    print("✅ Model loading successful!")
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped")

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

    return True


def main():
    """Main startup function."""
    print("🚀 SD Multi-Modal Platform - Direct Startup")
    print("Choose startup mode:")
    print("1. Minimal Mode (fastest, API testing only)")
    print("2. CPU Mode (slower, full functionality)")
    print("3. Auto-detect best mode")

    choice = input("\nEnter choice (1/2/3) or press Enter for auto: ").strip()

    if choice == "1":
        return start_with_minimal_mode()
    elif choice == "2":
        return start_with_cpu_mode()
    else:
        # Auto-detect
        print("\n🔍 Auto-detecting best startup mode...")

        # Check if models exist
        models_exist = Path("models").exists()

        # Check CUDA
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False

        print(f"Models available: {models_exist}")
        print(f"CUDA available: {cuda_available}")

        if not models_exist:
            print("➡️  No models found - using minimal mode")
            return start_with_minimal_mode()
        else:
            print("➡️  Models found - using CPU mode")
            return start_with_cpu_mode()


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
        sys.exit(0)
