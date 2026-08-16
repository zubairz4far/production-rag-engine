import hashlib
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.schemas import IngestResponse, QueryRequest, QueryResponse, RetrieveResponse
from app.services.chunker import chunk_pages
from app.services.document_loader import SUPPORTED_SUFFIXES, load_document
from app.services.rag import RAGService
from app.services.vector_store import VectorStore

router = APIRouter()


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(get_settings())


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    try:
        VectorStore(settings).client.get_collections()
        qdrant = "ok"
    except Exception as exc:  # noqa: BLE001 - health checks should report dependency failure
        qdrant = f"error: {type(exc).__name__}"
    return {"status": "ok", "service": settings.app_name, "qdrant": qdrant}


@router.post("/v1/documents/upload", response_model=IngestResponse)
def upload_document(file: Annotated[UploadFile, File()]) -> IngestResponse:
    settings = get_settings()
    name = Path(file.filename or "upload")
    if name.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
        )

    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / name.name
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    document_id = hashlib.sha256(target.read_bytes()).hexdigest()[:24]
    pages = load_document(target)
    chunks = chunk_pages(pages, settings.chunk_size_words, settings.chunk_overlap_words)
    indexed = VectorStore(settings).index_chunks(document_id, name.name, chunks)

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
