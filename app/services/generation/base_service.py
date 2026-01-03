"""
Compatibility shim.

Canonical implementations live under `services/` so sync API and workers share
the same behavior.
"""

from services.generation.base_service import *  # noqa: F401,F403
