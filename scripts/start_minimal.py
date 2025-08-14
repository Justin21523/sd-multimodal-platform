#!/usr/bin/env python3
"""
Minimal startup script for debugging Phase 3 issues.
Starts the API server without model initialization for testing.
"""
import logging
import sys
import os
import asyncio
from pathlib import Path
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set minimal environment variables
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("ENABLE_CPU_OFFLOAD", "true")
os.environ.setdefault("USE_ATTENTION_SLICING", "true")
os.environ.setdefault("DEBUG", "true")

from app.config import settings
from utils.logging_utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


async def create_minimal_app():
    """Create FastAPI app with minimal configuration."""
    try:
        # Import and create app
        from app.main import app

        # Override lifespan to skip model initialization
        @app.middleware("http")
        async def minimal_mode_middleware(request, call_next):
            # Add header to indicate minimal mode
            response = await call_next(request)
            response.headers["X-Minimal-Mode"] = "true"
            return response

        logger.info("✅ Minimal FastAPI app created successfully")
        return app

    except Exception as e:
        logger.error(f"❌ Failed to create minimal app: {e}")
        raise


def start_minimal_server():
    """Start server in minimal mode."""
    print("🚀 Starting SD Multi-Modal Platform in MINIMAL MODE")
    print("=" * 60)
    print("This mode starts the API server without model initialization")
    print("Useful for testing API structure and debugging startup issues")
    print("=" * 60)

    # Show configuration
    print(f"Host: {settings.HOST}")
    print(f"Port: {settings.PORT}")
    print(f"Device: {settings.DEVICE}")
    print(f"Debug: {settings.DEBUG}")
    print(f"API Docs: http://localhost:{settings.PORT}{settings.API_PREFIX}/docs")

    try:
        # Start server with minimal configuration
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=False,  # Disable reload to avoid import issues
            log_level="info",
            access_log=True,
        )

    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_minimal_server()
