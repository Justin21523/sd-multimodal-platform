"""
Request logging middleware.

The app also has a function-based middleware in `app/main.py` that attaches
request_id + timing headers. This middleware exists to keep imports stable and
provide a minimal extension point.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.config import settings
from utils.logging_utils import get_request_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = getattr(request.state, "request_id", None)
        if not request_id and getattr(settings, "ENABLE_REQUEST_ID", True):
            request_id = str(uuid.uuid4())[:8]
            request.state.request_id = request_id

        start = time.time()
        logger = get_request_logger(request_id or "request")
        logger.info(
            "Request start",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )

        response = await call_next(request)

        duration = time.time() - start
        response.headers.setdefault("X-Request-ID", request_id or "")
        response.headers.setdefault("X-Process-Time", f"{duration:.3f}s")

        logger.info(
            "Request end",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": duration,
            },
        )

        return response

