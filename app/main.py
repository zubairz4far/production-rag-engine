import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.observability import RequestObservabilityMiddleware
from app.core.rate_limit import RateLimitMiddleware

logging.basicConfig(level=logging.INFO, format="%(message)s")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.4.0",
    description="Hybrid RAG with citations, evaluation, observability, and public demo mode.",
)
app.add_middleware(RequestObservabilityMiddleware)
app.add_middleware(RateLimitMiddleware)
app.include_router(router)
