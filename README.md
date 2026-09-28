# Ally

RAG platform over confidential documents + agentic layer that acts on external
tools (Jira, Calendar, Drive), self-hosted LLM serving, hybrid retrieval, and
full production observability.

## Stack

| Concern              | Tool                                      |
|-----------------------|--------------------------------------------|
| Ingestion / OCR       | Unstructured.io, AWS Textract              |
| Workflow orchestration| Temporal                                   |
| Vector + hybrid search| Qdrant (dense + sparse)                    |
| Reranking             | bge-reranker-v2 / Cohere Rerank            |
| Retrieval eval        | Ragas (Recall@K, context precision)        |
| LLM serving           | vLLM (Qwen3-7B)                            |
| LLM routing/fallback  | LiteLLM                                    |
| Agent orchestration   | LangGraph                                  |
| Structured extraction | Instructor + Pydantic                      |
| API                   | FastAPI (REST + WebSocket)                 |
| Cache                 | Redis + RedisVL (semantic cache)           |
| Chat history          | MongoDB                                    |
| Chunk/metadata store  | Postgres                                   |
| Topic clustering      | BERTopic (UMAP + HDBSCAN + c-TF-IDF)       |
| Tracing/observability | Langfuse + OpenTelemetry                   |
| Resilience            | Tenacity (retries), pybreaker (circuit breaker) |

## Repo layout

```
ally/
├── services/
│   ├── ingestion/     # Temporal workflows + activities for OCR/ingest
│   ├── retrieval/     # Qdrant hybrid search, reranking
│   ├── chat/          # Prompt assembly w/ citations + MongoDB conversation history
│   ├── llm_gateway/   # LiteLLM router: Qwen3 (primary) + OpenAI (fallback), circuit breaker
│   ├── auth/          # Tenant provisioning + API key -> tenant resolution
│   ├── agent/         # LangGraph graph: intent -> RAG node | tool node
│   └── api/           # FastAPI app: REST + WebSocket chat
├── workflows/         # Temporal workflow/worker entrypoints
├── eval/              # Ragas eval sets + scripts (Recall@K, etc.)
├── clustering/        # BERTopic pipeline
├── infra/
│   ├── docker-compose.yml
│   └── postgres/      # Versioned SQL migrations
└── observability/     # Langfuse/OTel config
```

## Phase roadmap

- [x] Phase 0 — Infra bootstrap (this scaffold)
- [x] Phase 1 — Ingestion & OCR pipeline (Temporal + Unstructured/Textract)
- [x] Phase 2 — Hybrid retrieval + Ragas eval, measured on a 121-document labeled corpus (Recall@5 table below)
- [x] Phase 3 — LLM gateway + streaming chat with citations
- [ ] Phase 4 — LangGraph agent layer (RAG vs. tool actions)
- [ ] Phase 5 — Semantic + exact-match caching
- [ ] Phase 6 — BERTopic clustering
- [ ] Phase 7 — Observability & hardening

## Getting started

```bash
# 1. Bring up infra
cd infra
docker compose up -d

# If the API/worker image already exists, rebuild it after dependency or
# Dockerfile changes so the spaCy model is installed into the image:
# docker compose build --no-cache api worker

# 2. Verify services
#    Qdrant:      http://localhost:6333/dashboard
#    Temporal UI: http://localhost:8080
#    Langfuse:    http://localhost:3001
#    RedisInsight: http://localhost:8001

# 3. Install Python deps (in a venv)
cd ..
pip install -r requirements.txt

# 4. Provision a tenant and API key (all endpoints require one)
#    Prints the raw key once -- it is stored only as a SHA-256 hash.
python -m services.auth.provision_tenant create "My Org"

# 5. Backfill Phase 2 indexes for existing Postgres chunks
python services/retrieval/backfill_index.py

# 6. Run the API
uvicorn services.api.main:app --reload --port 8000

# 7. Run the ingestion worker (separate terminal)
python workflows/worker.py

# 8. Run the BM25 flusher (separate terminal, or as a cron job)
#    Full rebuilds are debounced during ingestion bursts; this catches up
#    whenever the index is left dirty.
python -m services.retrieval.bm25_rebuild_worker
```

## Authentication and multi-tenancy

Every document and chunk belongs to a tenant. Callers authenticate with an
API key, which resolves to the tenant that owns the data:

- REST: `X-API-Key: ally_...`
- WebSocket: `/ws/chat?api_key=ally_...` (browsers cannot set custom
  headers during the WebSocket handshake)

`tenant_id` is a required argument on `retrieve()`, `dense_search()`, and
`BM25Index.search()` -- not an optional filter -- so tenant scoping cannot
be omitted by accident.

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "X-API-Key: ally_your-key-here" \
  -F "file=@document.pdf"
```

## Chat: LLM gateway + streaming with citations (Phase 3)

### Running the primary model locally

The primary model is served over an OpenAI-compatible HTTP API, so any
server works. Set `QWEN_VLLM_BASE_URL` to point at it (default
`http://127.0.0.1:8001/v1`) and `QWEN_MODEL_NAME` to the model id that
server reports.

With vLLM (production, GPU):

```bash
vllm serve Qwen/Qwen3-7B --port 8001
```

With `llama.cpp` (flags depend on your hardware):

```bash
llama-server -hf Qwen/Qwen3-7B --port 8001 -c <context window> -np 1 -dev Vulkan1 -ngl 28
```

`/ws/chat` runs retrieve -> build prompt -> stream -> persist history on
every user message:

```
user message
   -> retrieve_async(query, tenant_id, top_k=CHAT_RETRIEVAL_TOP_K)   [services/retrieval/pipeline.py]
   -> numbered sources sent to the client                            [{"type": "sources", ...}]
   -> build_messages(history, sources, query)                        [services/chat/prompt.py]
   -> stream_chat_completion(messages)                                [services/llm_gateway/router.py]
   -> tokens streamed to the client as they arrive                   [{"type": "token", ...}]
   -> full answer + user turn persisted                              [services/chat/history.py]
   -> {"type": "done", "citations": sources}
```

**LLM gateway** (`services/llm_gateway/router.py`): a `litellm.Router`
with the self-hosted Qwen3 deployment (via vLLM's OpenAI-compatible
server, `QWEN_VLLM_BASE_URL`) as the primary model. When `OPENAI_API_KEY`
is set, `LLM_FALLBACK_MODEL` (default `gpt-4o-mini`) is registered as an
automatic fallback — Router retries the primary `LLM_MAX_RETRIES` times,
then fails over. A `pybreaker` circuit breaker sits in front of Router:
after `CIRCUIT_BREAKER_FAIL_MAX` consecutive total failures (every retry
and fallback exhausted), it opens for `CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS`
so a dead backend fails fast instead of every chat message paying the
full retry+fallback latency only to fail anyway.

**Citations**: `services/chat/prompt.py` numbers the retrieved chunks
(`[1]`, `[2]`, ...) and instructs the model to cite by that number for
every claim. The same numbered list — with `filename` and `page_number`
— is sent to the client as the `sources` message before any tokens
stream, and again as `citations` in the `done` message, so the client can
render inline `[1]`-style markers as links to a specific page without
needing any additional lookup.

**No-context fast path (just for now)**: when retrieval returns nothing,
the gateway is not called at all — an empty context block invites 
the model to answer from outside training knowledge despite 
the system prompt, and a known-empty-context answer doesn't need to be 
generated to know what it should say.

**Conversation history** (`services/chat/history.py`): stored in
MongoDB, scoped by `(tenant_id, conversation_id)`, trimmed to the last
`CHAT_HISTORY_MAX_MESSAGES` via an atomic `$push`/`$slice` rather than a
read-modify-write. A WebSocket connection gets a fresh `conversation_id`
unless the client passes `?conversation_id=...` to resume one.

**Follow-up turns queries**: Follow-up turns are condensed into 
a standalone retrieval query before hitting Qdrant/BM25 
(`services/chat/condense.py`) — e.g. "what about page 2?" is rewritten 
using prior turns before it's used for search. This only affects 
the retrieval query; the LLM prompt still receives the original message
and full history. Skipped on the first turn of a conversation, and fails
open (falls back to the raw query) if the condensation call itself fails.

## Benchmark corpus and retrieval numbers (Phase 2)

`eval/eval_set.json` labels a set of hand-written queries against their known-correct (document, page) pairs, verified against the tenant's already-ingested corpus. Only PDFs are used for page-level labels, since a page number isn't a meaningful concept in formats without real pagination.

Assumes the target tenant's documents are already ingested (via the normal upload/ingestion path — no separate seeding step).

```bash
# Measure Recall@5 per pipeline stage
python eval/run_ragas_eval.py --eval-set eval/eval_set.json --k 5
```

`eval/results.json` is overwritten wholesale by each run, and it carries the per-query rows behind the table below, so a regression can be traced to the queries that caused it rather than only to a moved average. Ragas' `context_precision` / `context_recall` (this benchmark is pending) only run when `OPENAI_API_KEY` is set; the Recall@K comparison needs no LLM.

### Measured Recall@5

k=5, tenant `00000000-...-000000000000`:

| Config | Recall@5 | All labels in top 5 | No label in top 5 |
| --- | --- | --- | --- |
| `vector_only` (dense Qdrant) | 0.92 | 44/50 | 2 |
| `hybrid` (dense + BM25, RRF) | 0.85 | 41/50 | 6 |
| `hybrid_reranked` (+ cross-encoder) | **0.96** | 46/50 | 0 |
| `hybrid_reranked_mmr` (+ MMR, diversity) | 0.88 | 39/50 | 1 |

**The full pipeline reaches 0.96 Recall@5, with all labels found in 46 of 50 queries and no query missing every label.** The cross-encoder reranker is the biggest contributor. It lifts the hybrid candidate list from 0.85 to 0.96 and above the dense-only baseline of 0.92.

MMR trades a little page recall for diversity: it drops near-duplicate chunks even when they are relevant, so its 0.88 isn't a regression. Page recall can't measure the redundancy MMR removes.

**Next optimization:** BM25 fusion currently scores below dense-only retrieval (0.85 vs 0.92). Tuning it, for example with lexical-only ablations or fusion weights, is the main remaining upside before the reranker step.

*Note: labels in `eval_set.json` are (document, page) pairs. If a labeled document is edited, re-ingested, or removed, they can drift out of sync with the corpus.*

## Tests

```bash
# Unit tests (fast, fully mocked, no services required)
python -m unittest discover -s tests -q

# Integration tests (require live Postgres + Qdrant; test_integration_retrieval
# also downloads the embedding/reranker models on first run)
cd infra && docker compose up -d postgres qdrant && cd ..
RUN_INTEGRATION_TESTS=1 python -m unittest discover -s tests -p "test_integration*.py"
```

## Environment variables

Copy `.env.example` to `.env` and fill in:
- `PROJECT_NAME` (defaults to `Ally`)
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `EMBEDDING_MODEL`
- `POSTGRES_DSN`, `MONGO_URI`, `MONGO_DB_NAME`, `REDIS_URL`, `TEMPORAL_ADDRESS`
- `BM25_INDEX_PATH`, `BM25_REBUILD_MIN_INTERVAL_SECONDS`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` / `TEXTRACT_S3_BUCKET` (for Textract)
- `QWEN_VLLM_BASE_URL`, `QWEN_MODEL_NAME` (self-hosted primary chat model)
- `OPENAI_API_KEY`, `LLM_FALLBACK_MODEL` (LLM gateway fallback)
- `LLM_MAX_RETRIES`, `LLM_REQUEST_TIMEOUT_SECONDS`, `CIRCUIT_BREAKER_FAIL_MAX`, `CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS`
- `CHAT_RETRIEVAL_TOP_K`, `CHAT_HISTORY_MAX_MESSAGES`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `MAX_UPLOAD_BYTES`
- `HF_TOKEN`
