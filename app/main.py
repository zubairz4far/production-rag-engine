import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.observability import RequestObservabilityMiddleware
from app.core.rate_limit import RateLimitMiddleware
from app.core.security import SecurityHeadersMiddleware, unhandled_exception_handler

logging.basicConfig(level=logging.INFO, format="%(message)s")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Hybrid RAG with citations, evaluation, observability, and public demo mode.",
)
app.add_exception_handler(Exception, unhandled_exception_handler)
app.add_middleware(RequestObservabilityMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.include_router(router)
