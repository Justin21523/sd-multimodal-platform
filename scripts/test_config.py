#!/usr/bin/env python3
# scripts/test_config.py
"""
Configuration test script for SD Multi-Modal Platform.
Tests configuration loading and validation.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_import():
    """Test basic configuration import."""
    try:
        from app.config import Settings, settings

        print("✅ Configuration import successful")
        print(f"   Settings type: {type(settings)}")
        print(f"   Pydantic version: {settings.model_config}")
        return True
    except Exception as e:
        print(f"❌ Configuration import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_environment_variables():
    """Test environment variable parsing."""
    try:
        # Test ALLOWED_ORIGINS parsing
        test_cases = [
            "http://localhost:3000,http://localhost:8080",
            "http://localhost:3000, http://localhost:8080",  # With spaces
            "http://localhost:3000",  # Single origin
            "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000",  # Multiple
        ]

        for test_origins in test_cases:
            print(f"🧪 Testing ALLOWED_ORIGINS: '{test_origins}'")

            # Set environment variable
            os.environ["ALLOWED_ORIGINS"] = test_origins

            # Import fresh settings
            from app.config import Settings

            test_settings = Settings()

            # Get parsed origins
            origins = test_settings.get_allowed_origins()
            print(f"   Parsed as: {origins}")

            # Validate
            assert isinstance(origins, list), f"Expected list, got {type(origins)}"
            assert all(
                isinstance(origin, str) for origin in origins
            ), "All origins should be strings"
            assert all(
                origin.strip() == origin for origin in origins
            ), "Origins should be stripped"
            assert len(origins) > 0, "Should have at least one origin"

            print(f"   ✅ Passed with {len(origins)} origins")

        return True

    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        return False


def test_pydantic_validation():
    """Test Pydantic validation."""
    try:
        from app.config import Settings

        # Test valid configuration
        valid_config = {
            "API_PREFIX": "/api/v1",
            "DEVICE": "cuda",
            "TORCH_DTYPE": "float16",
            "ALLOWED_ORIGINS": "http://localhost:3000,http://localhost:8080",
        }

        settings = Settings(**valid_config)  # type: ignore
        print("✅ Valid configuration accepted")
        print(f"   API prefix: {settings.API_PREFIX}")
        print(f"   Device: {settings.DEVICE}")
        print(f"   CORS origins: {settings.get_allowed_origins()}")

        # Test invalid configurations
        invalid_configs = [
            {
                "TORCH_DTYPE": "invalid_dtype",
                "ALLOWED_ORIGINS": "http://localhost:3000",
            },
            {"LOG_LEVEL": "INVALID_LEVEL", "ALLOWED_ORIGINS": "http://localhost:3000"},
        ]

        for i, invalid_config in enumerate(invalid_configs):
            try:
                Settings(**invalid_config)  # type: ignore
                print(f"⚠️  Invalid config {i+1} was unexpectedly accepted")
            except Exception as e:
                print(f"✅ Invalid config {i+1} correctly rejected: {type(e).__name__}")

        return True

    except Exception as e:
        print(f"❌ Pydantic validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_file_configuration():
    """Test configuration from .env file."""
    try:
        # Create temporary .env file
        test_env_content = """
# Test configuration
API_PREFIX=/api/v1
DEVICE=cuda
TORCH_DTYPE=float16
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000
LOG_LEVEL=INFO
USE_SDPA=true
ENABLE_XFORMERS=false
"""

        env_file = Path(".env.test")
        env_file.write_text(test_env_content.strip())

        try:
            # Test loading from file
            from app.config import Settings

            # Temporarily modify env_file
            original_env_file = getattr(Settings.model_config, "env_file", None)
            Settings.model_config["env_file"] = str(env_file)

            test_settings = Settings()

            # Validate loaded settings
            assert test_settings.API_PREFIX == "/api/v1"
            assert test_settings.DEVICE == "cuda"
            assert test_settings.TORCH_DTYPE == "float16"
            assert test_settings.LOG_LEVEL == "INFO"

            origins = test_settings.get_allowed_origins()
            assert len(origins) == 3
            assert "http://localhost:3000" in origins

            print("✅ File configuration loading successful")

            # Restore original env_file
            if original_env_file:
                Settings.model_config["env_file"] = original_env_file

            return True

        finally:
            # Cleanup
            if env_file.exists():
                env_file.unlink()

    except Exception as e:
        print(f"❌ File configuration test failed: {e}")
        return False


def test_cors_origins_methods():
    """Test CORS origins helper methods."""
    try:
        from app.config import Settings

        # Test with string input
        settings = Settings(
            ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8080"
        )
        origins = settings.get_allowed_origins()

        assert isinstance(origins, list)
        assert len(origins) == 2
        assert "http://localhost:3000" in origins
        assert "http://localhost:8080" in origins

        print("✅ CORS origins methods working correctly")
        return True

    except Exception as e:
        print(f"❌ CORS origins methods test failed: {e}")
        return False


def test_default_values():
    """Test default configuration values."""
    try:
        from app.config import Settings

        # Create settings with minimal input
        settings = Settings()

        # Check important defaults
        assert settings.API_PREFIX == "/api/v1"
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000
        assert settings.TORCH_DTYPE == "float16"
        assert settings.USE_SDPA == True
        assert settings.ENABLE_XFORMERS == False

        # Check CORS origins default
        origins = settings.get_allowed_origins()
        assert isinstance(origins, list)
        assert len(origins) >= 1

        print("✅ Default values are correct")
        return True

    except Exception as e:
        print(f"❌ Default values test failed: {e}")
        return False


def diagnostic_info():
    """Print diagnostic information."""
    print("\n" + "=" * 50)
    print("🔍 Diagnostic Information")
    print("=" * 50)

    # Python version
    print(f"Python: {sys.version}")

    # Working directory
    print(f"Working directory: {os.getcwd()}")

    # Environment variables
    print("Environment variables:")
    env_vars = [
        "ALLOWED_ORIGINS",
        "API_PREFIX",
        "DEVICE",
        "TORCH_DTYPE",
        "LOG_LEVEL",
        "USE_SDPA",
        "ENABLE_XFORMERS",
    ]

    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")

    # Check if .env file exists
    env_file = Path(".env")
    print(f".env file exists: {env_file.exists()}")

    if env_file.exists():
        print(f".env file size: {env_file.stat().st_size} bytes")

        # Show first few lines of .env file (excluding secrets)
        try:
            content = env_file.read_text()
            lines = content.split("\n")[:10]  # First 10 lines
            print(".env content (first 10 lines):")
            for line in lines:
                if not any(
                    secret in line.upper() for secret in ["SECRET", "KEY", "PASSWORD"]
                ):
                    print(f"  {line}")
                else:
                    print(f"  {line.split('=')[0]}=***")
        except Exception as e:
            print(f"Could not read .env file: {e}")

    # Pydantic version
    try:
        import pydantic

        print(f"Pydantic version: {pydantic.__version__}")
    except ImportError:
        print("Pydantic not installed")


def main():
    """Run all configuration tests."""
    print("🧪 SD Multi-Modal Platform - Configuration Test Suite")
    print("=" * 60)

    tests = [
        ("Basic Import", test_basic_import),
        ("Environment Variables", test_environment_variables),
        ("Pydantic Validation", test_pydantic_validation),
        ("File Configuration", test_file_configuration),
        ("CORS Origins Methods", test_cors_origins_methods),
        ("Default Values", test_default_values),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n🔄 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n🔍 Running diagnostics...")
        diagnostic_info()

        print("\n💡 Troubleshooting suggestions:")
        print("1. Check your .env file format")
        print("2. Verify ALLOWED_ORIGINS is comma-separated without spaces")
        print("3. Ensure Pydantic version is compatible")
        print("4. Try: pip install --upgrade pydantic pydantic-settings")

        return 1
    else:
        print("🎉 All configuration tests passed!")
        return 0


if __name__ == "__main__":
    exit(main())
