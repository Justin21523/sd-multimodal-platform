# utils/logging_utils.py 中的配置

from app.config import settings

config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "()": StructuredFormatter,  # 自定義 JSON 格式器
        },
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "simple",  # Development mode: human-readable format
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "structured",  # Production mode: JSON format
            "filename": settings.LOG_FILE,
            "maxBytes": 10485760,  # 10MB rotating file size
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {  # root logger
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn": {  # FastAPI Server Logging
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "diffusers": {  # AI model logging
            "level": "WARNING",  # reduce noise
            "handlers": ["file"],
            "propagate": False,
        },
    },
}
