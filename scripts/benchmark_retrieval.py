from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.services.chunker import chunk_pages
from app.services.document_loader import PageText
from app.services.reranker import Reranker
from app.services.vector_store import RetrievedChunk, VectorStore


DATASET = Path("evals/retrieval_benchmark.json")
TOP_K = 5


def reciprocal_rank(items: list[RetrievedChunk], expected_source: str) -> float:
    for rank, item in enumerate(items, start=1):
        if item.source == expected_source:
            return 1.0 / rank
    return 0.0


def hit_at_k(items: list[RetrievedChunk], expected_source: str) -> float:
    return float(any(item.source == expected_source for item in items[:TOP_K]))


def evaluate_mode(name: str, queries: list[dict], search_fn) -> dict:
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for row in queries:
        started = time.perf_counter()
        items = search_fn(row["query"])
        latencies_ms.append((time.perf_counter() - started) * 1000)
        hits.append(hit_at_k(items, row["expected_source"]))
        reciprocal_ranks.append(reciprocal_rank(items, row["expected_source"]))

    ordered_latency = sorted(latencies_ms)
    p95_index = max(0, min(len(ordered_latency) - 1, int(len(ordered_latency) * 0.95) - 1))

    return {
        "mode": name,
        "queries": len(queries),
        "hit_at_5": statistics.mean(hits),
        "mrr": statistics.mean(reciprocal_ranks),
        "mean_latency_ms": statistics.mean(latencies_ms),
        "p95_latency_ms": ordered_latency[p95_index],
    }


def main() -> None:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    settings = Settings(
        qdrant_url=":memory:",
        qdrant_collection="retrieval_benchmark",
        chunk_size_words=90,
        chunk_overlap_words=15,
        retrieval_prefetch=12,
    )

    client = QdrantClient(location=":memory:")
    store = VectorStore(settings, client=client)
    store.reset_collection()

    for index, document in enumerate(payload["documents"]):
        chunks = chunk_pages(
            [PageText(page=1, text=document["text"])],
            chunk_size_words=settings.chunk_size_words,
            overlap_words=settings.chunk_overlap_words,
        )
        store.index_chunks(
            document_id=f"benchmark-doc-{index}",
            source=document["source"],
            chunks=chunks,
        )

    queries = payload["queries"]
    reranker = Reranker(settings.reranker_model)

    def dense(query: str) -> list[RetrievedChunk]:
        return store.dense_search(query, TOP_K)

    def sparse(query: str) -> list[RetrievedChunk]:
        return store.sparse_search(query, TOP_K)

    def hybrid(query: str) -> list[RetrievedChunk]:
        return store.hybrid_search(query, TOP_K)

    def hybrid_rerank(query: str) -> list[RetrievedChunk]:
        candidates = store.hybrid_search(query, settings.retrieval_prefetch)
        return reranker.rerank(query, candidates, TOP_K)

    results = [
        evaluate_mode("dense", queries, dense),
        evaluate_mode("sparse_bm25", queries, sparse),
        evaluate_mode("hybrid_rrf", queries, hybrid),
        evaluate_mode("hybrid_rrf_rerank", queries, hybrid_rerank),
    ]

    print("\nRetrieval benchmark")
    print("=" * 86)
    print(f"{'mode':24} {'Hit@5':>10} {'MRR':>10} {'mean ms':>14} {'p95 ms':>14}")
    print("-" * 86)
    for row in results:
        print(
            f"{row['mode']:24} "
            f"{row['hit_at_5']:10.3f} "
            f"{row['mrr']:10.3f} "
            f"{row['mean_latency_ms']:14.2f} "
            f"{row['p95_latency_ms']:14.2f}"
        )

    Path("benchmark_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print("\nSaved machine-readable results to benchmark_results.json")


if __name__ == "__main__":
    main()
