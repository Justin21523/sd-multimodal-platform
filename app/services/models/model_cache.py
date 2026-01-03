"""
Compatibility shim.

Canonical implementations live under `services/` so sync API and workers share
the same behavior.
"""

from services.models.model_cache import *  # noqa: F401,F403
