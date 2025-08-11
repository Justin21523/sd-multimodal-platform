# utils/logging_utils.py
"""
Structured logging utilities with JSON formatting for production monitoring.
Phase 2: Backend Framework & Basic API Services
"""

import logging
import logging.config
import logging.handlers
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from app.config import settings


class StructuredFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Outputs logs in JSON format for easy parsing by monitoring systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""

        # Basic log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id  # type: ignore[attr-defined]

        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id  # type: ignore[attr-defined]

        if hasattr(record, "method"):
            log_entry["method"] = record.method  # type: ignore[attr-defined]

        if hasattr(record, "url"):
            log_entry["url"] = record.url  # type: ignore[attr-defined]

        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code  # type: ignore[attr-defined]

        if hasattr(record, "process_time"):
            log_entry["process_time"] = record.process_time  # type: ignore[attr-defined]

        if hasattr(record, "error_type"):
            log_entry["error_type"] = record.error_type  # type: ignore[attr-defined]

        if hasattr(record, "client_ip"):
            log_entry["client_ip"] = record.client_ip  # type: ignore[attr-defined]

        if hasattr(record, "user_agent"):
            log_entry["user_agent"] = record.user_agent  # type: ignore[attr-defined]

        # Add generation-specific fields
        if hasattr(record, "model_used"):
            log_entry["model_used"] = record.model_used  # type: ignore[attr-defined]

        if hasattr(record, "generation_time"):
            log_entry["generation_time"] = record.generation_time  # type: ignore[attr-defined]

        if hasattr(record, "vram_used"):
            log_entry["vram_used"] = record.vram_used  # type: ignore[attr-defined]

        if hasattr(record, "image_count"):
            log_entry["image_count"] = record.image_count  # type: ignore[attr-defined]

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for console output during development.
    """

    def __init__(self):
        # Color codes for different log levels
        self.COLORS = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
            "RESET": "\033[0m",  # Reset
        }

        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human reading with colors."""

        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Basic format
        parts = [
            f"{color}[{timestamp}]{reset}",
            f"{color}{record.levelname:8}{reset}",
            f"{record.name}:",
            record.getMessage(),
        ]

        # Add request ID if present
        if hasattr(record, "request_id"):
            parts.insert(-1, f"[{record.request_id}]")  # type: ignore[attr-defined]

        # Add processing time if present
        if hasattr(record, "process_time"):
            parts.append(f"({record.process_time:.3f}s)")  # type: ignore[attr-defined]

        return " ".join(parts)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json: bool = True,
) -> None:
    """
    Setup structured logging system with both console and file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, uses settings.LOG_FILE
        enable_json: Whether to use JSON formatting for file logs
    """

    # Use settings or defaults
    if log_level is None:
        log_level = getattr(settings, "LOG_LEVEL", "INFO")

    if log_file is None:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"  # type: ignore[attr-defined]

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper() if log_level else "INFO"))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with human-readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(
        getattr(logging, log_level.upper() if log_level else "INFO")
    )
    console_handler.setFormatter(HumanReadableFormatter())
    root_logger.addHandler(console_handler)

    # File handler with JSON format
    if enable_json:
        # Rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,  # type: ignore[arg-type]
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(
            getattr(logging, log_level.upper() if log_level else "INFO")
        )
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Set specific logger levels
    # Suppress overly verbose libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

    # Initial log message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized",
        extra={
            "log_level": log_level,
            "log_file": str(log_file),
            "json_enabled": enable_json,
        },
    )


def get_request_logger(request_id: str) -> logging.LoggerAdapter:
    """
    Get a logger adapter that automatically includes request ID in all log messages.

    Args:
        request_id: Unique request identifier

    Returns:
        LoggerAdapter with request_id in extra context
    """

    logger = logging.getLogger("request")

    class RequestLoggerAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            extra = kwargs.get("extra", {})
            extra["request_id"] = self.extra["request_id"] if self.extra else request_id
            kwargs["extra"] = extra
            return msg, kwargs

    return RequestLoggerAdapter(logger, {"request_id": request_id})


def get_generation_logger(
    request_id: str, model_name: str, task_type: str = "generation"
) -> logging.LoggerAdapter:
    """
    Get a logger adapter for AI generation tasks with context.

    Args:
        request_id: Unique request identifier
        model_name: Name of the AI model being used
        task_type: Type of generation task (txt2img, img2img, etc.)

    Returns:
        LoggerAdapter with generation context
    """

    logger = logging.getLogger("generation")

    class GenerationLoggerAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            extra = kwargs.get("extra", {})
            extra.update(self.extra)
            kwargs["extra"] = extra
            return msg, kwargs

    return GenerationLoggerAdapter(
        logger,
        {"request_id": request_id, "model_name": model_name, "task_type": task_type},
    )


# Utility functions for common logging patterns
def log_api_call(
    logger: logging.Logger,
    request_id: str,
    endpoint: str,
    method: str,
    status: str = "started",
):
    """Log API call with standard format."""
    logger.info(
        f"API call {status}",
        extra={
            "request_id": request_id,
            "endpoint": endpoint,
            "method": method,
            "status": status,
        },
    )


def log_generation_metrics(
    logger: logging.Logger,
    request_id: str,
    model_name: str,
    generation_time: float,
    image_count: int,
    vram_used: Optional[str] = None,
    prompt_length: Optional[int] = None,
):
    """Log generation performance metrics."""
    extra = {
        "request_id": request_id,
        "model_used": model_name,
        "generation_time": generation_time,
        "image_count": image_count,
    }

    if vram_used:
        extra["vram_used"] = vram_used
    if prompt_length:
        extra["prompt_length"] = prompt_length

    logger.info(
        f"Generation completed: {image_count} images in {generation_time:.2f}s",
        extra=extra,
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    request_id: str,
    context: Optional[Dict[str, Any]] = None,
):
    """Log error with full context information."""
    extra = {
        "request_id": request_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if context:
        extra.update(context)

    logger.error(f"Error occurred: {str(error)}", extra=extra, exc_info=True)
