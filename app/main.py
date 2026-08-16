import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.observability import RequestObservabilityMiddleware

logging.basicConfig(level=logging.INFO, format="%(message)s")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    description="Hybrid RAG with citations, evaluation, and production observability.",
)
app.add_middleware(RequestObservabilityMiddleware)
app.include_router(router)
