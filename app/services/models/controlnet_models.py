"""
Compatibility shim.

Canonical implementations live under `services/` so sync API and workers share
the same behavior.
"""

from services.models.controlnet_models import *  # noqa: F401,F403
