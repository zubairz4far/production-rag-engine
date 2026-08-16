from __future__ import annotations
import uuid
from dataclasses import dataclass
from qdrant_client import QdrantClient, models
from app.core.config import Settings
from app.services.chunker import Chunk


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    source: str
    page: int | None
    text: str
    score: float


class VectorStore:
    DENSE_VECTOR = "dense"
    SPARSE_VECTOR = "sparse"

    def __init__(self, settings: Settings):
        self.settings = settings
        kwargs = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        self.client = QdrantClient(**kwargs)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.settings.qdrant_collection):
            return
        self.client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config={
                self.DENSE_VECTOR: models.VectorParams(
                    size=self.client.get_embedding_size(self.settings.dense_model),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={self.SPARSE_VECTOR: models.SparseVectorParams()},
        )

    def index_chunks(self, document_id: str, source: str, chunks: list[Chunk]) -> int:
        self.ensure_collection()
        vectors, payloads, ids = [], [], []
        for chunk in chunks:
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk.chunk_index}:{chunk.page}"))
            vectors.append({
                self.DENSE_VECTOR: models.Document(text=chunk.text, model=self.settings.dense_model),
                self.SPARSE_VECTOR: models.Document(text=chunk.text, model=self.settings.sparse_model),
            })
            payloads.append({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "source": source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            })
            ids.append(chunk_id)
        if vectors:
            self.client.upload_collection(
                collection_name=self.settings.qdrant_collection,
                vectors=vectors,
                payload=payloads,
                ids=ids,
            )
        return len(chunks)

    def hybrid_search(self, query: str, limit: int) -> list[RetrievedChunk]:
        self.ensure_collection()
        prefetch_limit = max(limit, self.settings.retrieval_prefetch)
        points = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=self.settings.dense_model),
                    using=self.DENSE_VECTOR,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=self.settings.sparse_model),
                    using=self.SPARSE_VECTOR,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=prefetch_limit,
        ).points
        return [
            RetrievedChunk(
                chunk_id=str((p.payload or {}).get("chunk_id", p.id)),
                source=str((p.payload or {}).get("source", "unknown")),
                page=(p.payload or {}).get("page"),
                text=str((p.payload or {}).get("text", "")),
                score=float(p.score),
            )
            for p in points
        ]
