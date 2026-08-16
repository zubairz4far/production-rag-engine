from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from app.core.config import Settings
from app.services.chunker import chunk_pages
from app.services.document_loader import PageText
from app.services.reranker import Reranker
from app.services.vector_store import RetrievedChunk, VectorStore
from qdrant_client import QdrantClient


DEFAULT_DATASET = Path("evals/retrieval_benchmark.json")
TOP_K = 5


def expected_sources(row: dict) -> list[str]:
    if "expected_sources" in row:
        return list(row["expected_sources"])
    return [row["expected_source"]]


def reciprocal_rank(items: list[RetrievedChunk], expected: list[str]) -> float:
    wanted = set(expected)
    for rank, item in enumerate(items, start=1):
        if item.source in wanted:
            return 1.0 / rank
    return 0.0


def hit_at_k(items: list[RetrievedChunk], expected: list[str]) -> float:
    wanted = set(expected)
    return float(any(item.source in wanted for item in items[:TOP_K]))


def source_recall_at_k(items: list[RetrievedChunk], expected: list[str]) -> float:
    wanted = set(expected)
    if not wanted:
        return 1.0
    retrieved = {item.source for item in items[:TOP_K]}
    return len(wanted & retrieved) / len(wanted)


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return ordered[index]


def evaluate_mode(name: str, queries: list[dict], search_fn) -> dict:
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    source_recalls: list[float] = []
    latencies_ms: list[float] = []
    category_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"hit": [], "mrr": [], "source_recall": []}
    )

    for row in queries:
        expected = expected_sources(row)
        started = time.perf_counter()
        items = search_fn(row["query"])
        latencies_ms.append((time.perf_counter() - started) * 1000)

        hit = hit_at_k(items, expected)
        rr = reciprocal_rank(items, expected)
        recall = source_recall_at_k(items, expected)
        hits.append(hit)
        reciprocal_ranks.append(rr)
        source_recalls.append(recall)

        category = row.get("category", "uncategorized")
        category_scores[category]["hit"].append(hit)
        category_scores[category]["mrr"].append(rr)
        category_scores[category]["source_recall"].append(recall)

    by_category = {
        category: {
            "queries": len(values["hit"]),
            "hit_at_5": statistics.mean(values["hit"]),
            "mrr": statistics.mean(values["mrr"]),
            "source_recall_at_5": statistics.mean(values["source_recall"]),
        }
        for category, values in sorted(category_scores.items())
    }

    return {
        "mode": name,
        "queries": len(queries),
        "hit_at_5": statistics.mean(hits),
        "mrr": statistics.mean(reciprocal_ranks),
        "source_recall_at_5": statistics.mean(source_recalls),
        "mean_latency_ms": statistics.mean(latencies_ms),
        "p95_latency_ms": p95(latencies_ms),
        "by_category": by_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    config = payload.get("benchmark_config", {})
    top_k = int(config.get("top_k", TOP_K))

    settings = Settings(
        qdrant_url=":memory:",
        qdrant_collection="retrieval_benchmark",
        chunk_size_words=int(config.get("chunk_size_words", 90)),
        chunk_overlap_words=int(config.get("chunk_overlap_words", 15)),
        retrieval_prefetch=int(config.get("retrieval_prefetch", 12)),
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
        return store.dense_search(query, top_k)

    def sparse(query: str) -> list[RetrievedChunk]:
        return store.sparse_search(query, top_k)

    def hybrid(query: str) -> list[RetrievedChunk]:
        return store.hybrid_search(query, top_k)

    def hybrid_rerank(query: str) -> list[RetrievedChunk]:
        candidates = store.hybrid_search(query, settings.retrieval_prefetch)
        return reranker.rerank(query, candidates, top_k)

    results = [
        evaluate_mode("dense", queries, dense),
        evaluate_mode("sparse_bm25", queries, sparse),
        evaluate_mode("hybrid_rrf", queries, hybrid),
        evaluate_mode("hybrid_rrf_rerank", queries, hybrid_rerank),
    ]

    print(f"\n{payload.get('name', 'Retrieval benchmark')}")
    print(f"Dataset: {dataset_path} | documents={len(payload['documents'])} | queries={len(queries)}")
    print("=" * 104)
    print(
        f"{'mode':24} {'Hit@5':>10} {'MRR':>10} {'SrcR@5':>10} "
        f"{'mean ms':>14} {'p95 ms':>14}"
    )
    print("-" * 104)
    for row in results:
        print(
            f"{row['mode']:24} "
            f"{row['hit_at_5']:10.3f} "
            f"{row['mrr']:10.3f} "
            f"{row['source_recall_at_5']:10.3f} "
            f"{row['mean_latency_ms']:14.2f} "
            f"{row['p95_latency_ms']:14.2f}"
        )

    output = {
        "benchmark": payload.get("name", dataset_path.name),
        "dataset": str(dataset_path),
        "documents": len(payload["documents"]),
        "queries": len(queries),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved machine-readable results to {args.output}")


if __name__ == "__main__":
    main()
