# Production RAG Engine

A portfolio-grade Retrieval-Augmented Generation system built to demonstrate retrieval quality, reranking, grounded generation, evaluation, and production-style API design.

## Architecture

```text
Documents -> Loader -> Chunker -> Dense + BM25 indexing -> Qdrant
Query -> Dense retrieval + Sparse retrieval -> RRF fusion -> CrossEncoder reranker
      -> Context builder -> OpenAI-compatible LLM -> Answer + citations
      -> Evaluation: Hit@K, MRR, citation rate, answer recall, latency
```

## Stack

- FastAPI
- Qdrant
- FastEmbed
- Sentence Transformers
- CrossEncoder reranking
- OpenAI-compatible LLM endpoint
- Docker / Docker Compose

## Core features

- PDF, TXT, Markdown ingestion
- page-aware chunking
- dense semantic retrieval
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- second-stage reranking
- citation-aware grounded generation
- retrieval-only debugging endpoint
- golden evaluation dataset
- latency tracking

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

## Evaluation strategy

The repository includes a golden evaluation set under `evals/`. The target comparison is:

| Retrieval mode | Hit@5 | MRR | Answer quality | Latency |
|---|---:|---:|---:|---:|
| Dense only | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD |
| Hybrid + reranker | TBD | TBD | TBD | TBD |

Performance values will only be added after running real benchmarks.

## Roadmap

- [x] FastAPI service
- [x] document ingestion
- [x] hybrid retrieval
- [x] RRF fusion
- [x] CrossEncoder reranking
- [x] grounded generation
- [x] evaluation metrics
- [x] Docker setup
- [ ] run the first end-to-end benchmark
- [ ] expand evaluation set to 100+ questions
- [ ] add adversarial / prompt-injection tests
- [ ] add tracing and observability
- [ ] deploy a public demo
