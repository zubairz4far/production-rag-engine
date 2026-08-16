# Production RAG Engine

A portfolio-grade Retrieval-Augmented Generation system built to demonstrate retrieval quality, grounded generation, evaluation, observability, and production-style API design.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/zubairz4far/production-rag-engine)

## Architecture

```text
Documents -> Loader -> Chunker -> Dense + BM25 indexing -> Qdrant
Query -> Dense retrieval + Sparse retrieval -> RRF fusion -> optional CrossEncoder reranker
      -> Context builder -> OpenAI-compatible LLM -> Answer + citations
      -> Evaluation: Hit@K, MRR, source recall, citation precision/recall, refusal, injection resistance
      -> Observability: request IDs, structured logs, Prometheus metrics, latency histograms, readiness
```

## Stack

- FastAPI
- Qdrant
- FastEmbed
- dense semantic embeddings
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- optional CrossEncoder reranking
- OpenAI-compatible LLM endpoint
- Prometheus metrics
- structured JSON request logs
- Docker / Docker Compose
- GitHub Actions

## Core features

- PDF, TXT, Markdown ingestion
- page-aware chunking
- dense semantic retrieval
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- optional second-stage reranking
- citation-aware grounded generation
- retrieved-evidence prompt-injection hardening
- exact unsupported-question refusal contract
- retrieval-only debugging endpoint
- multi-document retrieval benchmarks
- category-level evaluation
- source recall for multi-source questions
- citation precision and recall scoring
- generation reliability benchmarks
- request ID propagation
- structured request telemetry
- Prometheus counters, gauges, and latency histograms
- liveness and dependency-aware readiness endpoints
- public portfolio demo UI
- public request rate limiting
- upload size guardrail
- non-root production container
- automated CI including Docker image build

## Public demo deployment

The repository includes a Render Blueprint (`render.yaml`) for a one-click public portfolio deployment. Render runs the Docker image with `DEMO_MODE=true`, a 30 requests/minute per-process rate limit, and the `/health/live` health check.

The public demo deliberately uses a lightweight seeded retriever so it can run on constrained free hosting while preserving the same `/v1/query`, `/v1/retrieve`, citation, evidence, and timing response contracts as the full service. The landing page clearly labels this mode. The measured production architecture remains the Qdrant + dense + BM25 + RRF engine documented below.

In public demo mode:

- `GET /` serves the interactive recruiter-facing UI.
- `POST /v1/query` returns grounded answers, citations, evidence, and timings.
- unsupported questions use the exact refusal contract.
- document upload is disabled to avoid implying persistence on ephemeral free hosting.
- `/docs`, `/metrics`, and health endpoints remain available.

For the full engine, keep `DEMO_MODE=false` and configure Qdrant plus the OpenAI-compatible model endpoint.

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

Then open:

- Demo / landing page: `http://localhost:8000/`
- API docs: `http://localhost:8000/docs`
- Qdrant dashboard: `http://localhost:6333/dashboard`
- Prometheus: `http://localhost:9090`
- Metrics endpoint: `http://localhost:8000/metrics`

To run only the lightweight public demo behavior locally, set `DEMO_MODE=true` before starting the API.

## Production health and observability

The service exposes separate process and dependency health endpoints:

- `GET /health/live` — process liveness; does not depend on Qdrant.
- `GET /health/ready` — readiness; returns HTTP 503 when Qdrant is unavailable in full mode and reports the demo store ready in demo mode.
- `GET /health` — compatibility alias for readiness.
- `GET /metrics` — Prometheus exposition endpoint.

Every HTTP request gets an `x-request-id`. If the caller supplies one it is propagated; otherwise the API creates a UUID. Structured JSON request logs include request ID, route, method, status, and elapsed time.

Prometheus instrumentation includes HTTP request counts, in-flight requests, HTTP latency, retrieval latency, generation latency, retrieved chunk count, and dependency readiness. Docker Compose includes a Prometheus service using `ops/prometheus.yml` to scrape the API every 15 seconds.

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

The repository contains two retrieval benchmark suites. Both run with an in-memory Qdrant instance, so no external database is required.

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

The practical default from this experiment is therefore **hybrid RRF without reranking for latency-sensitive retrieval**, with reranking reserved for workloads where top-rank precision matters more than latency. Production config follows that result with reranking disabled by default.

## Generation Reliability Benchmarks

Generation evaluation is intentionally separated from retrieval evaluation.

V1 contains 12 targeted cases covering grounded factual answers, exact unsupported-question refusal, stale-versus-current policy resolution, unresolved contradictory evidence, citation behavior, and malicious instructions embedded in retrieved documents.

V2 expands the suite to **40 adversarial cases across 10 categories**, including single- and multi-source grounding, partial support, current-vs-legacy resolution, unresolved conflicts, prompt injection, citation laundering, numeric reasoning, and scope/tier ambiguity.

```bash
# Offline CI-safe contract validation
python scripts/benchmark_generation.py \
  --dataset evals/generation_reliability_v2.json \
  --output generation_benchmark_v2_contract.json

# Local open-model evaluation
python scripts/benchmark_generation_local.py \
  --dataset evals/generation_reliability_v2.json \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --output generation_benchmark_v2_smollm2_360m.json

# Live configured OpenAI-compatible endpoint
python scripts/benchmark_generation.py \
  --dataset evals/generation_reliability_v2.json \
  --output generation_benchmark_v2_live.json \
  --live
```

Offline mode validates the benchmark and prompt-safety contract only. It does **not** fabricate model-quality results. Live/local scoring includes keyword recall, refusal accuracy, citation coverage, citation-ID validity, citation precision, citation recall, forbidden-term leakage, prompt-injection leakage, category pass rate, and overall strict pass rate.

Retrieved evidence is explicitly treated as **untrusted data** and wrapped in evidence delimiters. Instructions found inside documents are not allowed to override the system prompt.

## Evaluation philosophy

This repository treats RAG as an evaluated system rather than a framework demo. A more complex retrieval or generation design is only adopted when measurements justify the latency, reliability, and operational cost.

## Roadmap

- [x] FastAPI service
- [x] document ingestion
- [x] dense retrieval baseline
- [x] sparse BM25 baseline
- [x] hybrid retrieval
- [x] RRF fusion
- [x] optional CrossEncoder reranking
- [x] grounded generation
- [x] Docker setup
- [x] GitHub Actions CI
- [x] 20-question V1 retrieval benchmark
- [x] 120-question hard V2 retrieval benchmark
- [x] category-level retrieval evaluation
- [x] multi-source source-recall metric
- [x] publish measured V1 + V2 retrieval results
- [x] Generation Reliability Benchmark V1 dataset
- [x] Generation Reliability Benchmark V2 40-case adversarial dataset
- [x] citation validity, precision, recall, and refusal scoring
- [x] prompt-injection retrieval tests
- [x] harden retrieved evidence as untrusted input
- [x] run first local open-model generation baseline
- [x] request ID propagation and structured request logs
- [x] Prometheus request/retrieval/generation metrics
- [x] liveness and dependency-aware readiness probes
- [x] Docker Compose Prometheus collector
- [x] recruiter-facing public demo UI
- [x] one-click Render Blueprint
- [x] non-root container and Docker build CI gate
- [x] public rate limiting and upload size guardrail
- [ ] publish measured 40-case V2 generation results
- [ ] add claim-level citation entailment scoring
- [ ] add distributed tracing export (OpenTelemetry)
- [ ] approve one-click deployment in a Render account
