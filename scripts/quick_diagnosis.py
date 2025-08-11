#!/usr/bin/env python3
"""
Quick diagnosis script for configuration issues.
Helps identify Pydantic v2 configuration problems.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("🔍 SD Multi-Modal Platform - Quick Diagnosis")
print("=" * 50)

# Step 1: Check Python and package versions
print("1. 📦 Package Versions:")
try:
    import pydantic

    print(f"   Pydantic: {pydantic.__version__}")
except ImportError:
    print("   ❌ Pydantic not installed")

try:
    import pydantic_settings

    print(f"   Pydantic Settings: {pydantic_settings.__version__}")
except ImportError:
    print("   ❌ pydantic-settings not installed")

try:
    import torch

    print(f"   PyTorch: {torch.__version__}")
except ImportError:
    print("   ❌ PyTorch not installed")

# Step 2: Check environment variables
print("\n2. 🌍 Environment Variables:")
env_vars = ["ALLOWED_ORIGINS", "API_PREFIX", "DEVICE", "TORCH_DTYPE"]
for var in env_vars:
    value = os.environ.get(var, "Not set")
    print(f"   {var}: {value}")

# Step 3: Check .env file
print("\n3. 📄 .env File:")
env_file = Path(".env")
if env_file.exists():
    print(f"   ✅ .env exists ({env_file.stat().st_size} bytes)")

    try:
        content = env_file.read_text()
        lines = [
            line.strip()
            for line in content.split("\n")
            if line.strip() and not line.startswith("#")
        ]
        print(f"   📝 Found {len(lines)} configuration lines")

        # Check for ALLOWED_ORIGINS specifically
        for line in lines:
            if line.startswith("ALLOWED_ORIGINS"):
                print(f"   🎯 ALLOWED_ORIGINS line: {line}")
                break
        else:
            print("   ⚠️  No ALLOWED_ORIGINS found in .env")

    except Exception as e:
        print(f"   ❌ Error reading .env: {e}")
else:
    print("   ⚠️  .env file not found")

# Step 4: Test basic imports
print("\n4. 🔬 Import Tests:")

# Test pydantic v2 syntax
try:
    from pydantic import Field, field_validator, computed_field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    print("   ✅ Pydantic v2 imports successful")
except ImportError as e:
    print(f"   ❌ Pydantic v2 import failed: {e}")

# Test minimal configuration
print("\n5. 🧪 Minimal Configuration Test:")
try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class TestSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            extra="ignore",  # 忽略沒有在模型定義的鍵
            case_sensitive=False,  # 可選：與 .env 大小寫無關
        )
        ALLOWED_ORIGINS: str = Field(default="http://localhost:3000")
        API_PREFIX: str = Field(default="/api/v1")

    test_settings = TestSettings()
    print(f"   ✅ Minimal config works")
    print(f"   📋 ALLOWED_ORIGINS: {test_settings.ALLOWED_ORIGINS}")
    print(f"   📋 API_PREFIX: {test_settings.API_PREFIX}")

except Exception as e:
    print(f"   ❌ Minimal config failed: {e}")
    import traceback

    traceback.print_exc()

# Step 6: Test our actual configuration
print("\n6. 🎯 Full Configuration Test:")
try:
    # Clear any problematic env vars first
    problematic_vars = ["ALLOWED_ORIGINS"]
    for var in problematic_vars:
        if var in os.environ:
            print(f"   🧹 Clearing {var} from environment")
            del os.environ[var]

    from app.config import Settings, settings

    print("   ✅ Configuration import successful")
    print(f"   📋 Device: {settings.DEVICE}")
    print(f"   📋 API Prefix: {settings.API_PREFIX}")
    print(f"   📋 Allowed Origins: {settings.get_allowed_origins()}")

except Exception as e:
    print(f"   ❌ Configuration import failed: {e}")
    import traceback

    traceback.print_exc()

# Step 7: Suggestions
print("\n7. 💡 Troubleshooting Suggestions:")

# Check if running from correct directory
if not Path("app").exists():
    print("   ⚠️  Run from project root directory (where 'app' folder exists)")

# Check .env format
if env_file.exists():
    content = env_file.read_text()
    if '"' in content and "ALLOWED_ORIGINS" in content:
        print("   ⚠️  Remove quotes from ALLOWED_ORIGINS in .env file")

    if ", " in content and "ALLOWED_ORIGINS" in content:
        print("   ⚠️  Remove spaces after commas in ALLOWED_ORIGINS")

# Version suggestions
try:
    import pydantic

    major, minor = map(int, pydantic.__version__.split(".")[:2])
    if major < 2:
        print(f"   ⚠️  Upgrade Pydantic: pip install --upgrade 'pydantic>=2.0'")
except:
    pass

print("\n8. 🔧 Quick Fix Commands:")
print("   # Update packages:")
print("   pip install --upgrade 'pydantic>=2.0' pydantic-settings")
print()
print("   # Fix .env format:")
print("   # Ensure ALLOWED_ORIGINS has no quotes and no spaces after commas")
print("   # Example: ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080")
print()
print("   # Test configuration:")
print("   python scripts/test_config.py")
