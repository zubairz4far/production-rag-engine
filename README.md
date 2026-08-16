# Production RAG Engine

**v1.0.0** — an evaluated Retrieval-Augmented Generation system built to demonstrate retrieval quality, grounded generation, adversarial evaluation, observability, deployment hardening, and explicit failure analysis.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/zubairz4far/production-rag-engine)

## Architecture

```text
Documents -> Loader -> Page-aware chunker -> Dense + BM25 indexing -> Qdrant
Query -> Dense retrieval + Sparse retrieval -> RRF fusion -> optional CrossEncoder reranker
      -> Evidence builder -> OpenAI-compatible LLM -> Grounded answer + citations
      -> Evaluation: Hit@K, MRR, source recall, citation precision/recall,
                     refusal behavior, prompt-injection leakage, latency
      -> Operations: request IDs, structured logs, Prometheus metrics,
                     readiness/liveness, rate limiting, hardened Docker runtime
```

## Stack

- FastAPI
- Qdrant
- FastEmbed
- `sentence-transformers/all-MiniLM-L6-v2` dense retrieval
- `Qdrant/bm25` sparse retrieval
- Reciprocal Rank Fusion
- optional `Xenova/ms-marco-MiniLM-L-6-v2` CrossEncoder reranking
- OpenAI-compatible generation endpoint
- Prometheus
- Docker / Docker Compose
- GitHub Actions

## What this project demonstrates

- PDF, TXT, and Markdown ingestion
- page-aware chunking
- dense semantic retrieval
- BM25 sparse retrieval
- hybrid RRF retrieval
- optional second-stage reranking
- source-aware evidence objects and citations
- exact unsupported-question refusal contract
- retrieved-context prompt-injection hardening
- 20-query retrieval smoke benchmark
- 120-query hard retrieval benchmark
- 12-case Generation Reliability V1 suite
- 40-case adversarial Generation Reliability V2 suite
- category-level failure analysis
- request ID propagation and structured telemetry
- Prometheus counters, gauges, and latency histograms
- liveness and dependency-aware readiness
- public recruiter-facing demo mode
- non-root container and Docker build CI gate
- application rate limiting and upload size guardrails
- security headers and sanitized 500 responses
- Dependabot and least-privilege GitHub Actions permissions

## Public demo deployment

The repository includes `render.yaml` and a one-click Render Blueprint.

The public deployment uses `DEMO_MODE=true`. It deliberately swaps the full embedding/Qdrant stack for a small seeded deterministic retriever so the portfolio demo remains lightweight while preserving the same `/v1/query`, `/v1/retrieve`, evidence, citation, refusal, and timing response contracts.

The landing page clearly labels this distinction. The measured production architecture remains the Qdrant + dense + BM25 + RRF engine documented below.

In demo mode:

- `GET /` serves the interactive portfolio UI.
- `POST /v1/query` returns answer, evidence, citations, and timings.
- unsupported questions use the exact refusal contract.
- document upload is disabled so ephemeral hosting is not presented as persistent storage.
- `/docs`, `/metrics`, `/health/live`, and `/health/ready` remain available.

For the full engine, keep `DEMO_MODE=false` and configure Qdrant plus an OpenAI-compatible model endpoint.

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

Open:

- Demo / landing page: `http://localhost:8000/`
- API docs: `http://localhost:8000/docs`
- Qdrant dashboard: `http://localhost:6333/dashboard`
- Prometheus: `http://localhost:9090`
- Metrics: `http://localhost:8000/metrics`

To run only the lightweight demo behavior locally, set `DEMO_MODE=true`.

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

## Production hardening

### Health and telemetry

- `GET /health/live` checks process liveness only.
- `GET /health/ready` checks Qdrant in full mode and returns HTTP 503 when unavailable.
- dependency error types are not exposed to callers.
- every request receives an `x-request-id`; caller-supplied IDs are propagated.
- structured logs include request ID, route, method, status, duration, Render request ID, and Cloudflare `CF-Ray` when present.
- `/metrics` exposes request counts, in-flight requests, HTTP latency, retrieval latency, generation latency, retrieved chunk count, and dependency readiness.

### Public API protection

- sliding-window rate limiting applies to `/v1/*` endpoints.
- the default rate-limit identity is the socket peer, so arbitrary `X-Forwarded-For` values are not trusted.
- `CLIENT_IP_HEADER` can be configured only when deployment is behind an edge proxy that overwrites the chosen header.
- the Render Blueprint uses `cf-connecting-ip`.
- stale rate-limit buckets are periodically removed to bound per-process state.
- uploads are capped at 10 MB by default and partial oversized uploads are deleted.

### HTTP and container hardening

Responses include:

- `Cache-Control: no-store`
- Content Security Policy with `frame-ancestors 'none'`
- `Permissions-Policy`
- `Referrer-Policy: no-referrer`
- HSTS
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`

Unexpected exceptions return a generic HTTP 500 body with a request ID instead of internal exception details.

The production container:

- runs as an unprivileged user,
- has a healthcheck,
- honors the platform `$PORT`,
- disables the Uvicorn `Server` header,
- disables duplicate access logging in favor of structured application request logs,
- excludes `.env`, Git metadata, tests, eval artifacts, caches, and local uploads from the Docker build context.

The in-process limiter is intentionally a single-instance protection layer. A scaled multi-replica deployment should replace it with a gateway or shared-store rate limiter.

## Retrieval benchmarks

Both suites run against an in-memory Qdrant instance, so the benchmark does not require an external database.

```bash
pip install -e .

python scripts/benchmark_retrieval.py \
  --dataset evals/retrieval_benchmark.json \
  --output benchmark_results_v1.json

python scripts/benchmark_retrieval.py \
  --dataset evals/retrieval_benchmark_v2.json \
  --output benchmark_results_v2.json
```

### Retrieval V1 — 20-query smoke suite

| Retrieval mode | Hit@5 | MRR | Source recall@5 | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|
| Dense only | 1.000 | 0.975 | 1.000 | 23.35 ms | 23.81 ms |
| Sparse BM25 | 1.000 | **1.000** | 1.000 | **2.36 ms** | **2.62 ms** |
| Hybrid RRF | 1.000 | 0.975 | 1.000 | 25.83 ms | 26.21 ms |
| Hybrid RRF + reranker | 1.000 | **1.000** | 1.000 | 200.25 ms | 223.34 ms |

V1 is intentionally terminology-heavy and functions as a smoke benchmark rather than the main retrieval claim.

### Retrieval V2 — 24 documents / 120 hard queries

V2 includes semantic paraphrases, lexical distractors, scope/tier ambiguity, indirect numeric questions, version conflicts, contrast queries, and multi-source retrieval.

| Retrieval mode | Hit@5 | MRR | Source recall@5 | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.983 | 0.869 | 0.975 | 23.48 ms | 23.92 ms |
| Sparse BM25 | 0.983 | 0.880 | 0.983 | **2.11 ms** | **2.83 ms** |
| **Hybrid RRF** | **1.000** | 0.908 | **1.000** | 28.88 ms | 34.08 ms |
| **Hybrid RRF + reranker** | **1.000** | **0.925** | **1.000** | 425.09 ms | 454.58 ms |

### Retrieval conclusion

Hybrid RRF is the default production tradeoff in this experiment. It achieves perfect Hit@5 and source recall across all 120 V2 queries without the reranker's large latency penalty. Reranking raises overall MRR from 0.908 to 0.925 and semantic-paraphrase MRR from 0.903 to 0.984, but increases mean retrieval latency from about 29 ms to 425 ms.

These are measurements on the repository's synthetic policy benchmark, not a claim of universal retrieval superiority.

## Generation Reliability Benchmarks

Generation is evaluated independently from retrieval so model failure cannot be hidden behind retrieval quality.

V1 contains 12 cases. V2 contains 40 adversarial cases across:

- supported single-source grounding
- supported multi-source grounding
- unsupported refusal
- partial support
- current-versus-legacy evidence
- unresolved conflicts
- prompt injection inside retrieved evidence
- citation laundering
- numeric reasoning
- scope/tier ambiguity

```bash
python scripts/benchmark_generation.py \
  --dataset evals/generation_reliability_v2.json \
  --output generation_benchmark_v2_contract.json

python scripts/benchmark_generation_local.py \
  --dataset evals/generation_reliability_v2.json \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --output generation_benchmark_v2_smollm2_360m.json
```

### Measured local baseline

Model: `HuggingFaceTB/SmolLM2-360M-Instruct`

| Suite | Cases passed | Strict pass rate | Citation-ID validity | Prompt-injection leak rate | Mean generation |
|---|---:|---:|---:|---:|---:|
| Generation V1 | 2 / 12 | 16.7% | 100% | 0% | 2507 ms |
| Generation V2 | 8 / 40 | 20.0% | 100% | 0% | 2625 ms |

The V2 result is intentionally reported as a **model-capacity failure baseline**, not a success metric. The 360M model strongly over-refuses: it passes all 4 unsupported-refusal cases and all 4 conservative partial-support cases, but fails the supported-answer, conflict-resolution, citation-laundering, numeric-reasoning, and scope-resolution categories by returning the refusal text instead of using available evidence.

Therefore, the zero observed prompt-injection leakage must not be interpreted as strong grounded-generation capability—the model often avoids the attack by refusing the supported task entirely.

The service remains model-agnostic through an OpenAI-compatible endpoint, allowing stronger models to be evaluated with the same reliability suite without changing retrieval or scoring logic.

The exact measured summary is committed at `evals/results/latest_summary.json`, sourced from GitHub Actions run `31963034743` on commit `8019fa9a6ff13a092ce7feb890b425e1aa1bedeb`.

## Evaluation philosophy

This project treats RAG as an evaluated system rather than a framework demo. Complexity is added only when measurements justify the reliability and latency cost. Weak model results are kept and explained rather than removed from the portfolio.

## CI and maintenance

GitHub Actions gates changes with:

1. package installation
2. Ruff linting
3. unit and regression tests
4. production Docker image build
5. retrieval V1 + V2 benchmarks
6. Generation Reliability V1 + V2 contract validation
7. local open-model generation benchmarks
8. machine-readable benchmark artifact upload

Workflow tokens explicitly use read-only repository permissions. Dependabot monitors Python dependencies, the Docker base image, and GitHub Actions dependencies weekly.

## v1.0.0 completion status

- [x] full FastAPI RAG API
- [x] document ingestion and chunking
- [x] dense + sparse + hybrid retrieval
- [x] optional reranking
- [x] citation-constrained generation
- [x] adversarial generation evaluation
- [x] measured retrieval V1/V2 results
- [x] measured generation V1/V2 baseline with failure analysis
- [x] durable benchmark summary committed to the repository
- [x] Prometheus observability and structured tracing
- [x] liveness and readiness probes
- [x] public demo mode and recruiter UI
- [x] one-click Render Blueprint
- [x] rate limiting and upload guardrails
- [x] security headers and safe error responses
- [x] non-root hardened Docker runtime
- [x] CI lint/tests/Docker build/benchmark gates
- [x] least-privilege workflow permissions
- [x] Dependabot dependency monitoring

The repository is code-complete for the portfolio. Activating the public URL only requires approving the prepared Blueprint in a Render account.
