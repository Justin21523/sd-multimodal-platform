"""
Simple in-memory rate limiting middleware.

Notes:
- This is a lightweight guardrail for API abuse during development.
- Queue-level rate limiting is also implemented in `app/core/queue_manager.py`.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, DefaultDict, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.config import settings


@dataclass
class _Window:
    timestamps: Deque[float]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token-bucket-ish sliding window limiter:
    - `RATE_LIMIT_REQUESTS` per `RATE_LIMIT_MINUTES`
    - + `RATE_LIMIT_BURST` extra allowance (short spikes)
    """

    def __init__(self, app):
        super().__init__(app)
        self.enabled = bool(getattr(settings, "ENABLE_RATE_LIMITING", True))
        self.limit = int(getattr(settings, "RATE_LIMIT_REQUESTS", 100))
        self.window_sec = int(getattr(settings, "RATE_LIMIT_MINUTES", 1)) * 60
        self.burst = int(getattr(settings, "RATE_LIMIT_BURST", 10))
        self._requests: DefaultDict[str, _Window] = defaultdict(
            lambda: _Window(timestamps=deque(maxlen=self.limit + self.burst + 10))
        )

    def _key(self, request: Request) -> str:
        # Prefer API key if present, else client IP.
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key:{api_key}"
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _prune(self, window: _Window, now: float) -> None:
        cutoff = now - self.window_sec
        while window.timestamps and window.timestamps[0] < cutoff:
            window.timestamps.popleft()

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Do not rate limit health checks by default.
        if request.url.path.endswith("/health") or request.url.path.endswith(
            "/health/simple"
        ):
            return await call_next(request)

        now = time.time()
        key = self._key(request)
        window = self._requests[key]
        self._prune(window, now)

        allowed = self.limit + self.burst
        if len(window.timestamps) >= allowed:
            retry_after = 1
            if window.timestamps:
                oldest = window.timestamps[0]
                retry_after = max(1, int(self.window_sec - (now - oldest)))

            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        window.timestamps.append(now)
        return await call_next(request)

