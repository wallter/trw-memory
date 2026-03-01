"""X-API-Key authentication middleware."""

from __future__ import annotations

import os

from fastapi import Request, Response
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse

_SKIP_PATHS = {"/v1/health", "/health", "/openapi.json", "/docs", "/redoc"}
_API_KEY_HEADER = "X-API-Key"
_ENV_VAR = "MEMORY_API_KEY"


async def api_key_middleware(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Validate X-API-Key header against MEMORY_API_KEY env var.

    Auth is disabled when the env var is not set.  Health, docs, and
    OpenAPI endpoints are always exempt.
    """
    expected = os.environ.get(_ENV_VAR)
    if expected is None:
        # No key configured = auth disabled
        return await call_next(request)
    if request.url.path in _SKIP_PATHS:
        return await call_next(request)
    provided = request.headers.get(_API_KEY_HEADER)
    if not provided or provided != expected:
        return JSONResponse(
            status_code=401, content={"detail": "Invalid or missing API key"}
        )
    return await call_next(request)
