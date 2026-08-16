from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.services.vector_store import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str):
        self.model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        texts = [item.text for item in candidates]
        scores = list(self.model.rerank(query, texts))

        reranked = [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                source=item.source,
                page=item.page,
                text=item.text,
                score=float(score),
            )
            for item, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
