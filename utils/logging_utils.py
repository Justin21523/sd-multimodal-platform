# utils/logging_utils.py
"""
SD Multi-Modal Platform - Logging Utilities
Phase 1: Structured Logging and Request Tracking
"""

import logging
import logging.config
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from app.config import settings


class StructuredFormatter(logging.Formatter):
    """Structured logging formatter for JSON output"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add additional fields if available
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "task_id"):
            log_entry["task_id"] = record.task_id
        if hasattr(record, "model"):
            log_entry["model"] = record.model
        if hasattr(record, "generation_time"):
            log_entry["generation_time"] = record.generation_time
        if hasattr(record, "vram_used"):
            log_entry["vram_used"] = record.vram_used

        # Handle exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging():
    """Initialize global logging configuration"""

    # Make sure log directory exists
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "structured",
                "filename": settings.LOG_FILE,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {  # root logger
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn": {"level": "INFO", "handlers": ["console"], "propagate": False},
            "diffusers": {"level": "WARNING", "handlers": ["file"], "propagate": False},
        },
    }

    logging.config.dictConfig(config)

    # Initialize request logger
    logger = logging.getLogger(__name__)
    logger.info("✅ Logging system initialized")


def get_request_logger(request_id: str) -> logging.Logger:
    """Get a logger instance with request context"""
    logger = logging.getLogger("request")
    return logging.LoggerAdapter(logger, {"request_id": request_id})
