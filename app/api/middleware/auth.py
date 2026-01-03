"""
Optional API key authentication middleware.

If `API_KEYS` is provided (env var), requests must include `X-API-Key`.
Format accepted:
- JSON array string: ["key1","key2"]
- Comma-separated: key1,key2
"""

from __future__ import annotations

import json
import os
from typing import List

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


def _load_api_keys() -> List[str]:
    raw = os.getenv("API_KEYS", "").strip()
    if not raw:
        return []

    # JSON list
    if raw.startswith("["):
        try:
            values = json.loads(raw)
            return [str(v).strip() for v in values if str(v).strip()]
        except Exception:
            return []

    # Comma-separated
    return [v.strip() for v in raw.split(",") if v.strip()]


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.api_keys = set(_load_api_keys())

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.api_keys:
            return await call_next(request)

        # Allow health checks without auth (operational convenience)
        if request.url.path.endswith("/health") or request.url.path.endswith(
            "/health/simple"
        ):
            return await call_next(request)

        api_key = request.headers.get("x-api-key", "")
        if api_key not in self.api_keys:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": "Unauthorized",
                    "error_code": "UNAUTHORIZED",
                },
            )

        return await call_next(request)

