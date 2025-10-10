"""
Base service with shared functionality for all generation services
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from app.shared_cache import shared_cache
from app.config import settings

logger = logging.getLogger(__name__)


class BaseGenerationService:
    """Base service with common generation functionality"""

    def __init__(self):
        self.output_dir = self._get_output_directory()

    def _get_output_directory(self) -> str:
        """Get output directory for generated files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project = getattr(self, "project_name", "generation")
        output_path = os.path.join(
            shared_cache.cache_root, "outputs", project, "images", timestamp
        )
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def _save_output_metadata(self, filename: str, metadata: Dict[str, Any]):
        """Save metadata for generated output"""
        metadata_file = os.path.splitext(filename)[0] + ".json"

        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.debug(f"Metadata saved to {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate generation parameters against limits"""
        # Check steps
        if parameters.get("steps", 0) > settings.MAX_STEPS:
            raise ValueError(f"Steps cannot exceed {settings.MAX_STEPS}")

        # Check CFG scale
        if parameters.get("cfg_scale", 0) > settings.MAX_CFG:
            raise ValueError(f"CFG scale cannot exceed {settings.MAX_CFG}")

        # Check resolution
        width = parameters.get("width", 512)
        height = parameters.get("height", 512)
        if width * height > settings.MAX_PIXELS:
            raise ValueError(
                f"Resolution {width}x{height} exceeds maximum pixels {app_settings.MAX_PIXELS}"
            )

        return True

    def _create_task_metadata(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create standard task metadata"""
        return {
            "timestamp": datetime.now().isoformat(),
            "service": self.__class__.__name__,
            "parameters": parameters,
            "device": str(getattr(self, "device", "cpu")),
            "version": getattr(settings, "APP_VERSION", "0.1.0"),
        }
