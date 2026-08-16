# Production RAG Engine

A portfolio-grade Retrieval-Augmented Generation system built to demonstrate retrieval quality, reranking, grounded generation, evaluation, and production-style API design.

## Architecture

```text
Documents -> Loader -> Chunker -> Dense + BM25 indexing -> Qdrant
Query -> Dense retrieval + Sparse retrieval -> RRF fusion -> FastEmbed CrossEncoder reranker
      -> Context builder -> OpenAI-compatible LLM -> Answer + citations
      -> Evaluation: Hit@K, MRR, citation rate, answer recall, latency
```

## Stack

- FastAPI
- Qdrant
- FastEmbed
- dense semantic embeddings
- BM25 sparse retrieval
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
- multi-document golden retrieval benchmark
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

## Reproducible retrieval benchmark

The first benchmark uses six competing documents and 20 labeled questions. It runs Qdrant in-memory, so no external database is required.

```bash
pip install -e .
python scripts/benchmark_retrieval.py
```

The script compares four systems:

1. dense-only retrieval
2. sparse BM25 retrieval
3. dense + sparse with Reciprocal Rank Fusion
4. hybrid RRF + CrossEncoder reranking

It reports `Hit@5`, `MRR`, mean latency, and P95 latency, and writes the machine-readable output to `benchmark_results.json`.

The same benchmark runs in GitHub Actions and uploads the result JSON as a workflow artifact.

## Measured results

GitHub Actions benchmark, 20 labeled queries:

| Retrieval mode | Hit@5 | MRR | Mean latency | P95 latency |
|---|---:|---:|---:|---:|
| Dense only | 1.000 | 0.975 | 21.18 ms | 21.64 ms |
| Sparse BM25 | 1.000 | **1.000** | **2.56 ms** | **2.64 ms** |
| Hybrid RRF | 1.000 | 0.975 | 24.23 ms | 24.72 ms |
| Hybrid RRF + reranker | 1.000 | **1.000** | 178.66 ms | 191.85 ms |

### What this first experiment tells us

On this small, terminology-heavy corpus, BM25 is already extremely strong: it reaches perfect MRR while being much faster than the neural alternatives. The reranker also reaches perfect MRR, but increases mean retrieval latency from roughly 24 ms for hybrid RRF to roughly 179 ms.

That means the correct engineering conclusion is **not** that more complex retrieval is automatically better. The next benchmark must be harder and more semantic before choosing hybrid + reranking as the default production path.

This is why the project keeps dense, sparse, hybrid, and reranked retrieval as independently measurable configurations.

## Evaluation strategy

The repository separates retrieval evaluation from generation evaluation. Retrieval is compared experimentally first; grounded answer quality and citation behavior are measured after a retrieval configuration is selected.

The current 20-question suite is a smoke benchmark, not a final quality claim. The next evaluation set will contain 100+ questions with paraphrases, lexical mismatch, distractor passages, near-duplicate evidence, and adversarial cases.

## Roadmap

- [x] FastAPI service
- [x] document ingestion
- [x] dense retrieval baseline
- [x] sparse BM25 baseline
- [x] hybrid retrieval
- [x] RRF fusion
- [x] CrossEncoder reranking
- [x] grounded generation
- [x] evaluation metrics
- [x] Docker setup
- [x] 20-question multi-document retrieval benchmark
- [x] GitHub Actions benchmark workflow
- [x] publish first measured benchmark results
- [ ] expand evaluation set to 100+ harder questions
- [ ] add adversarial / prompt-injection tests
- [ ] measure grounded generation and citation correctness
- [ ] add tracing and observability
- [ ] deploy a public demo
