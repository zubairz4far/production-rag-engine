from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-process fixed-window limiter for public demo protection.

    Production multi-replica deployments should replace this with a shared limiter
    backed by an API gateway or Redis.
    """

    def __init__(self, app):
        super().__init__(app)
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if request.url.path.startswith(("/health", "/metrics")):
            return await call_next(request)

        forwarded = request.headers.get("x-forwarded-for", "")
        client_ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
        now = time.monotonic()
        cutoff = now - 60.0
        bucket = self.requests[client_ip]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= settings.rate_limit_per_minute:
            retry_after = max(1, int(60 - (now - bucket[0])))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."},
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        return await call_next(request)
