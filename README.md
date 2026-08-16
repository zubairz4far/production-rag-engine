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

The benchmark uses six competing documents and 20 labeled questions. It runs Qdrant in-memory, so no external database is required.

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

| Retrieval mode | Hit@5 | MRR | Mean latency | P95 latency |
|---|---:|---:|---:|---:|
| Dense only | pending benchmark | pending | pending | pending |
| Sparse BM25 | pending benchmark | pending | pending | pending |
| Hybrid RRF | pending benchmark | pending | pending | pending |
| Hybrid RRF + reranker | pending benchmark | pending | pending | pending |

No performance value is added here until it is produced by the automated benchmark.

## Evaluation strategy

The repository separates retrieval evaluation from generation evaluation. Retrieval is compared experimentally first; grounded answer quality and citation behavior are measured after a retrieval configuration is selected.

This makes it possible to attribute gains or regressions to the retrieval layer rather than hiding them inside an end-to-end LLM score.

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
- [ ] publish first measured benchmark results
- [ ] expand evaluation set to 100+ questions
- [ ] add adversarial / prompt-injection tests
- [ ] add tracing and observability
- [ ] deploy a public demo
