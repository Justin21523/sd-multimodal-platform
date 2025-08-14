#!/usr/bin/env python3
"""
Phase 4 testing and validation script
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_phase4_components():
    """Test Phase 4 components"""
    print("🧪 Testing Phase 4 Components...")

    # Test asset manager
    from services.assets.asset_manager import get_asset_manager

    asset_manager = get_asset_manager()
    success = await asset_manager.initialize()
    print(f"  Asset Manager: {'✅' if success else '❌'}")

    # Test ControlNet manager
    from services.processors.controlnet_service import get_controlnet_manager

    controlnet_manager = get_controlnet_manager()
    status = controlnet_manager.get_status()
    print(f"  ControlNet Manager: ✅ ({len(status['supported_types'])} types)")

    print("✅ Phase 4 component tests completed!")


if __name__ == "__main__":
    asyncio.run(test_phase4_components())
