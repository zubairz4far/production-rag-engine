from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import get_settings


def client_identity(request: Request, trusted_header: str | None) -> str:
    """Return a rate-limit key from a configured trusted proxy header or socket peer."""
    if trusted_header:
        value = request.headers.get(trusted_header, "").strip()
        if value:
            return value.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-process sliding-window limiter for public API protection.

    Multi-replica deployments should replace this with an API-gateway or shared-store limiter.
    Configure ``CLIENT_IP_HEADER`` only for a header overwritten by a trusted edge proxy.
    """

    def __init__(self, app):
        super().__init__(app)
        self.requests: dict[str, deque[float]] = defaultdict(deque)
        self.last_cleanup = time.monotonic()

    def _cleanup(self, now: float) -> None:
        if now - self.last_cleanup < 60.0:
            return
        cutoff = now - 60.0
        for key, bucket in list(self.requests.items()):
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if not bucket:
                del self.requests[key]
        self.last_cleanup = now

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        settings = get_settings()
        now = time.monotonic()
        self._cleanup(now)
        cutoff = now - 60.0
        key = client_identity(request, settings.client_ip_header)
        bucket = self.requests[key]
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
