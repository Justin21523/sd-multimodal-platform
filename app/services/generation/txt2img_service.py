"""
Compatibility shim.

Canonical implementations live under `services/` so sync API and workers share
the same behavior.
"""

from services.generation.txt2img_service import *  # noqa: F401,F403
