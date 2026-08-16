# Production RAG Engine

A portfolio-grade Retrieval-Augmented Generation system built to demonstrate retrieval quality, reranking, grounded generation, evaluation, and production-style API design.

## Architecture

```text
Documents -> Loader -> Chunker -> Dense + BM25 indexing -> Qdrant
Query -> Dense retrieval + Sparse retrieval -> RRF fusion -> FastEmbed CrossEncoder reranker
      -> Context builder -> OpenAI-compatible LLM -> Answer + citations
      -> Evaluation: Hit@K, MRR, source recall, citation rate, answer recall, latency
```

## Stack

- FastAPI
- Qdrant
- FastEmbed
- dense semantic embeddings
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- CrossEncoder reranking
- OpenAI-compatible LLM endpoint
- Docker / Docker Compose
- GitHub Actions

## Core features

- PDF, TXT, Markdown ingestion
- page-aware chunking
- dense semantic retrieval
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- second-stage reranking
- citation-aware grounded generation
- retrieval-only debugging endpoint
- multi-document retrieval benchmarks
- category-level evaluation
- source recall for multi-source questions
- latency tracking
- automated CI

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

Then open:

- API docs: `http://localhost:8000/docs`
- Qdrant dashboard: `http://localhost:6333/dashboard`

## API

### Upload a document

```bash
curl -X POST "http://localhost:8000/v1/documents/upload" \
  -F "file=@your_document.pdf"
```

### Retrieve evidence

```bash
curl -X POST "http://localhost:8000/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say?","top_k":5}'
```

### Generate a grounded answer

```bash
curl -X POST "http://localhost:8000/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say?","top_k":5}'
```

## Reproducible retrieval benchmarks

The repository contains two benchmark suites. Both run with an in-memory Qdrant instance, so no external database is required.

```bash
pip install -e .

# V1 smoke benchmark
python scripts/benchmark_retrieval.py \
  --dataset evals/retrieval_benchmark.json \
  --output benchmark_results_v1.json

# V2 hard benchmark
python scripts/benchmark_retrieval.py \
  --dataset evals/retrieval_benchmark_v2.json \
  --output benchmark_results_v2.json
```

The runner compares four systems:

1. dense-only retrieval
2. sparse BM25 retrieval
3. dense + sparse with Reciprocal Rank Fusion
4. hybrid RRF + CrossEncoder reranking

Metrics include `Hit@5`, `MRR`, `source recall@5`, mean latency, P95 latency, and per-category quality. GitHub Actions runs both suites and uploads machine-readable results.

## Benchmark V1 — smoke test

V1 contains six competing documents and 20 labeled questions.

| Retrieval mode | Hit@5 | MRR | Mean latency | P95 latency |
|---|---:|---:|---:|---:|
| Dense only | 1.000 | 0.975 | 23.32 ms | 24.07 ms |
| Sparse BM25 | 1.000 | **1.000** | **2.27 ms** | **2.35 ms** |
| Hybrid RRF | 1.000 | 0.975 | 25.71 ms | 25.98 ms |
| Hybrid RRF + reranker | 1.000 | **1.000** | 198.97 ms | 218.70 ms |

V1 is intentionally simple. BM25 performs extremely well because the corpus is terminology-heavy.

## Benchmark V2 — hard retrieval suite

V2 contains **24 near-overlapping policy documents and 120 labeled queries** across semantic paraphrases, lexical distractors, scope/tier ambiguity, numeric inference, version conflicts, contrast questions, and multi-source retrieval.

| Retrieval mode | Hit@5 | MRR | Source recall@5 | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.983 | 0.869 | 0.975 | 23.02 ms | 23.32 ms |
| Sparse BM25 | 0.983 | 0.880 | 0.983 | **1.99 ms** | **2.77 ms** |
| **Hybrid RRF** | **1.000** | 0.908 | **1.000** | 26.41 ms | 26.62 ms |
| **Hybrid RRF + reranker** | **1.000** | **0.925** | **1.000** | 427.11 ms | 450.72 ms |

### V2 findings

The harder benchmark changes the conclusion from V1:

- **Hybrid RRF is the strongest latency/coverage tradeoff.** It is the only non-reranked system with perfect Hit@5 and perfect source recall across all 120 cases.
- **Reranking improves ordering quality.** Hybrid + reranker raises overall MRR from 0.908 to 0.925.
- The reranker is especially useful for **semantic paraphrases**, where MRR rises from 0.903 for hybrid RRF to **0.984**.
- BM25 remains extremely fast, but semantic paraphrases expose its weakness: category Hit@5 falls to **0.935**.
- Dense retrieval handles paraphrases better than BM25 on coverage, but lexical distractors hurt its ordering quality.
- Reranking has a real cost: mean retrieval latency increases from about **26 ms to 427 ms**.

The practical default from this experiment is therefore **hybrid RRF without reranking for latency-sensitive retrieval**, with reranking reserved for workloads where top-rank precision matters more than latency.

## Evaluation philosophy

This repository treats RAG as an evaluated system rather than a framework demo. A more complex retrieval stack is only adopted when measurements justify the latency and operational cost.

Retrieval evaluation is kept separate from generation evaluation. The next stage measures whether the generation layer:

- answers only from retrieved evidence,
- cites the correct source,
- refuses unsupported questions,
- handles contradictory/versioned evidence,
- resists prompt injection inside retrieved documents.

## Roadmap

- [x] FastAPI service
- [x] document ingestion
- [x] dense retrieval baseline
- [x] sparse BM25 baseline
- [x] hybrid retrieval
- [x] RRF fusion
- [x] CrossEncoder reranking
- [x] grounded generation
- [x] Docker setup
- [x] GitHub Actions CI
- [x] 20-question V1 benchmark
- [x] 120-question hard V2 benchmark
- [x] category-level retrieval evaluation
- [x] multi-source source-recall metric
- [x] publish measured V1 + V2 results
- [ ] add grounded generation benchmark
- [ ] add citation correctness scoring
- [ ] add unsupported-question refusal tests
- [ ] add prompt-injection / adversarial retrieval tests
- [ ] add tracing and observability
- [ ] deploy a public demo
