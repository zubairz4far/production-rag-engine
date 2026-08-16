import hashlib
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from app.api.demo_ui import demo_page
from app.core.config import get_settings
from app.core.observability import DEPENDENCY_READY, metrics_response
from app.models.schemas import IngestResponse, QueryRequest, QueryResponse, RetrieveResponse
from app.services.chunker import chunk_pages
from app.services.demo_rag import DemoRAGService
from app.services.document_loader import SUPPORTED_SUFFIXES, load_document
from app.services.rag import RAGService

router = APIRouter()


@lru_cache
def get_rag_service() -> RAGService | DemoRAGService:
    settings = get_settings()
    return DemoRAGService() if settings.demo_mode else RAGService(settings)


@router.get("/", include_in_schema=False)
def root() -> Response:
    return demo_page()


@router.get("/health/live")
def liveness() -> dict:
    settings = get_settings()
    return {"status": "ok", "service": settings.app_name, "demo_mode": settings.demo_mode}


@router.get("/health/ready")
def readiness(response: Response) -> dict:
    settings = get_settings()
    if settings.demo_mode:
        DEPENDENCY_READY.labels("demo_store").set(1)
        return {
            "status": "ready",
            "service": settings.app_name,
            "demo_mode": True,
            "retriever": "seeded-lightweight",
        }

    try:
        service = get_rag_service()
        assert isinstance(service, RAGService)
        service.vector_store.client.get_collections()
        DEPENDENCY_READY.labels("qdrant").set(1)
        return {"status": "ready", "service": settings.app_name, "qdrant": "ok"}
    except Exception as exc:  # noqa: BLE001 - readiness reports dependency failure
        DEPENDENCY_READY.labels("qdrant").set(0)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "service": settings.app_name,
            "qdrant": f"error: {type(exc).__name__}",
        }


@router.get("/health")
def health(response: Response) -> dict:
    return readiness(response)


@router.get("/metrics")
def metrics() -> Response:
    return metrics_response()


@router.post("/v1/documents/upload", response_model=IngestResponse)
def upload_document(file: Annotated[UploadFile, File()]) -> IngestResponse:
    settings = get_settings()
    if settings.demo_mode:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document upload is disabled in public demo mode. Run the full engine locally for ingestion.",
        )

    name = Path(file.filename or "upload")
    if name.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / name.name
    written = 0
    try:
        with target.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Upload exceeds {settings.max_upload_mb} MB limit.",
                    )
                out.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise

    document_id = hashlib.sha256(target.read_bytes()).hexdigest()[:24]
    pages = load_document(target)
    chunks = chunk_pages(pages, settings.chunk_size_words, settings.chunk_overlap_words)
    service = get_rag_service()
    assert isinstance(service, RAGService)
    indexed = service.vector_store.index_chunks(document_id, name.name, chunks)

    return IngestResponse(
        document_id=document_id,
        source=name.name,
        chunks_indexed=indexed,
    )


@router.post("/v1/retrieve", response_model=RetrieveResponse)
def retrieve(request: QueryRequest) -> RetrieveResponse:
    return get_rag_service().retrieve(request.question, request.top_k)


@router.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return get_rag_service().query(request.question, request.top_k)
