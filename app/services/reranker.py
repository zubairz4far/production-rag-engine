from sentence_transformers import CrossEncoder
from app.services.vector_store import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self.model.predict([(query, item.text) for item in candidates])
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
