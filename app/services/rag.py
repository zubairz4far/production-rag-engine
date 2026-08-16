import time

from app.core.config import Settings
from app.models.schemas import Evidence, QueryResponse, RetrieveResponse
from app.services.llm import LLMClient
from app.services.reranker import Reranker
from app.services.vector_store import RetrievedChunk, VectorStore


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vector_store = VectorStore(settings)
        self.reranker = Reranker(settings.reranker_model)
        self.llm = LLMClient(settings)

    def _retrieve_chunks(
        self,
        question: str,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], float]:
        started = time.perf_counter()
        candidates = self.vector_store.hybrid_search(
            question,
            max(top_k, self.settings.retrieval_prefetch),
        )
        reranked = self.reranker.rerank(question, candidates, top_k)
        return reranked, (time.perf_counter() - started) * 1000

    @staticmethod
    def _evidence(chunks: list[RetrievedChunk]) -> list[Evidence]:
        return [
            Evidence(
                citation_id=f"C{i}",
                chunk_id=item.chunk_id,
                source=item.source,
                page=item.page,
                score=item.score,
                text=item.text,
            )
            for i, item in enumerate(chunks, 1)
        ]

    def retrieve(self, question: str, top_k: int) -> RetrieveResponse:
        chunks, retrieval_ms = self._retrieve_chunks(question, top_k)
        return RetrieveResponse(
            question=question,
            evidence=self._evidence(chunks),
            retrieval_ms=round(retrieval_ms, 2),
        )

    def query(self, question: str, top_k: int) -> QueryResponse:
        total_started = time.perf_counter()
        chunks, retrieval_ms = self._retrieve_chunks(question, top_k)

        generation_started = time.perf_counter()
        answer = self.llm.answer(question, chunks)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        return QueryResponse(
            question=question,
            evidence=self._evidence(chunks),
            retrieval_ms=round(retrieval_ms, 2),
            answer=answer,
            generation_ms=round(generation_ms, 2),
            total_ms=round((time.perf_counter() - total_started) * 1000, 2),
        )
