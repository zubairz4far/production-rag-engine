from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("rag.observability")

REQUESTS = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)
IN_FLIGHT = Gauge("rag_http_in_flight_requests", "Current in-flight HTTP requests")
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval latency in seconds",
)
GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds",
    "Generation latency in seconds",
)
RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Number of chunks returned by retrieval",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
)
DEPENDENCY_READY = Gauge(
    "rag_dependency_ready",
    "Dependency readiness state (1 ready, 0 not ready)",
    ["dependency"],
)


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or request.url.path


def log_event(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, separators=(",", ":"), default=str))


class RequestObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        IN_FLIGHT.inc()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - started
            path = _route_label(request)
            REQUESTS.labels(request.method, path, str(status)).inc()
            REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
            IN_FLIGHT.dec()
            log_event(
                "http_request",
                request_id=request_id,
                render_request_id=request.headers.get("rndr-id"),
                cf_ray=request.headers.get("cf-ray"),
                method=request.method,
                path=path,
                status=status,
                duration_ms=round(elapsed * 1000, 2),
            )


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
