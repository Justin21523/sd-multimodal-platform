# utils/logging_utils.py
"""
Structured logging utilities with JSON formatting for production monitoring.
Fixed structured logging utilities with proper generation logger
"""

import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
import json
import sys
import time
from datetime import datetime
import uuid
from pathlib import Path
from typing import Dict, Any, Optional


from app.config import settings


class JSONFormatter(logging.Formatter):
    """JSON 格式的日誌格式化器"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加異常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # 添加額外字段
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)  # type: ignore

        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """獲取日誌記錄器"""
    return logging.getLogger(name)


class StructuredLogger:
    """結構化日誌記錄器"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_with_context(self, level: str, message: str, **context):
        """帶上下文的日誌記錄"""
        extra = {"extra_fields": context}
        getattr(self.logger, level.lower())(message, extra=extra)

    def log_task_start(self, task_id: str, task_type: str, user_id: str = None):  # type: ignore
        """記錄任務開始"""
        self.log_with_context(
            "info",
            f"Task started: {task_id}",
            task_id=task_id,
            task_type=task_type,
            user_id=user_id,
            event_type="task_start",
        )

    def log_task_complete(self, task_id: str, duration: float, success: bool = True):
        """記錄任務完成"""
        self.log_with_context(
            "info" if success else "error",
            f"Task {'completed' if success else 'failed'}: {task_id}",
            task_id=task_id,
            duration=duration,
            success=success,
            event_type="task_complete",
        )

    def log_performance(self, operation: str, duration: float, **metrics):
        """記錄性能指標"""
        self.log_with_context(
            "info",
            f"Performance: {operation}",
            operation=operation,
            duration=duration,
            event_type="performance",
            **metrics,
        )


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

        # Add custom fields from extra
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for console output during development.
    """

    def __init__(self):
        super().__init__()
        self.fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        self.datefmt = "%H:%M:%S"
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

        # Add extra fields to message if present
        extra_fields = parts.copy()
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "asctime",
            ]:
                if key == "request_id":
                    extra_fields.append(f"req:{value}")
                elif key == "task_id":
                    extra_fields.append(f"task:{value}")
                elif key == "generation_time":
                    extra_fields.append(f"gen:{value}s")
                elif key == "vram_used":
                    extra_fields.append(f"vram:{value}")

        if extra_fields:
            record.msg = f"{record.msg} [{' | '.join(extra_fields)}]"

        return super().format(record)


def setup_logging(
    log_level: Optional[str] = "INFO",
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

    # Use settings or defaults
    if log_level is None:
        log_level = getattr(settings, "LOG_LEVEL", "INFO")

    if log_file is None:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"  # type: ignore[attr-defined]

    # File handler with JSON format
    try:
        file_handler = RotatingFileHandler(
            log_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5  # type: ignore[arg-type]
        )  # 10MB
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    except Exception as e:
        root_logger.warning(f"Failed to setup file logging: {e}")

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


class RequestLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically adds request_id to all log entries"""

    def __init__(self, logger: logging.Logger, request_id: str):
        super().__init__(logger, {"request_id": request_id})

    def process(self, msg, kwargs):
        # Ensure request_id is always included
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"].update(self.extra)  # type: ignore
        return msg, kwargs


class GenerationLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for generation tasks with model and task information"""

    def __init__(
        self, logger: logging.Logger, task_type: str, model_name: str = "unknown"
    ):
        extra = {
            "task_type": task_type,
            "model_name": model_name,
            "task_id": str(uuid.uuid4())[:8],
        }
        super().__init__(logger, extra)

    def process(self, msg, kwargs):
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"].update(self.extra)  # type: ignore
        return msg, kwargs


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
    request_id: Optional[str], model_name: str, task_type: str = "generation"
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


# Initialize logging when module is imported
setup_logging()
