# Ally

Multi-tenant RAG platform over confidential documents: OCR ingestion,
hybrid (dense + lexical) retrieval, a self-hosted-first LLM gateway with
streaming chat and page-level citations, and a LangGraph agent layer that
routes between answering from documents and acting on external tools —
Jira, Google Drive, Google Calendar, and the user's own PC via a local
companion agent (Phases 0-4, built).

Planned: semantic caching (Phase 5), topic clustering (Phase 6), and
full tracing/observability (Phase 7).
Multi-tenant RAG platform over confidential documents: OCR ingestion,
hybrid (dense + lexical) retrieval, a self-hosted-first LLM gateway with
streaming chat and page-level citations, and a LangGraph agent layer that
routes between answering from documents and acting on external tools —
Jira, Google Drive, Google Calendar, and the user's own PC via a local
companion agent (Phases 0-4, built).

Planned: semantic caching (Phase 5), topic clustering (Phase 6), and
full tracing/observability (Phase 7).

## Stack

Reflects what's actually imported and running today. Each row links to
the module that uses it.

| Concern                | Tool                                         | Where |
|-------------------------|-----------------------------------------------|-------|
| Ingestion / OCR         | Upload/storage adapters plus Temporal activities (Unstructured.io, AWS Textract) | `services/ingestion/`, `workflows/activities.py` |
| Workflow orchestration  | Temporal                                      | `workflows/` |
| Vector + hybrid search  | Qdrant (dense) + BM25 (`rank_bm25`), fused via RRF | `services/retrieval/` |
| Reranking               | bge-reranker-v2 (cross-encoder)               | `services/retrieval/reranker.py` |
| Retrieval eval          | Ragas (Recall@K, context precision/recall)    | `eval/run_ragas_eval.py` |
| LLM serving             | vLLM (Qwen3), OpenAI-compatible               | `services/llm_gateway/router.py` |
| LLM routing/fallback    | LiteLLM `Router`                              | `services/llm_gateway/router.py` |
| Circuit breaking        | pybreaker                                      | `services/llm_gateway/router.py` |
| Retries (transient DB)  | Tenacity                                       | `services/auth/api_keys.py` |
| Agent orchestration     | LangGraph (`StateGraph`)                       | `services/agent/graph.py` |
| Credential encryption   | `cryptography` (Fernet)                        | `services/agent/connected_accounts.py` |
| API                     | FastAPI (REST + WebSocket)                     | `services/api/main.py` |
| Chat history            | MongoDB                                        | `services/chat/mongo_history.py` |
| Chunk/metadata/auth/vault | Postgres                                    | `infra/postgres/` |
| Auth                    | API keys, SHA-256 hashed, per-tenant           | `services/auth/` |
| Local PC control         | WebSocket bridge + standalone companion agent | `services/agent/pc_agent_bridge.py`, `local_agent/` |

**Planned, not yet built** (Phases 5-7) — pinned in `requirements.txt`
ahead of the work landing, not imported anywhere yet:

| Concern              | Planned tool                              | Phase |
|-----------------------|--------------------------------------------|-------|
| Semantic/exact cache  | Redis + RedisVL                            | 5 |
| Topic clustering      | BERTopic (UMAP + HDBSCAN + c-TF-IDF)       | 6 |
| Tracing/observability | Langfuse + OpenTelemetry                   | 7 |

**Deferred by design, not forgotten**: `instructor` is pinned but not
adopted for Phase 4's intent classifier — its documented integration
patterns assume direct access to `litellm`'s completion function or an
OpenAI-SDK-shaped client, neither of which matches this project's
`litellm.Router`-based gateway. See `services/agent/classifier.py`'s
module docstring for the reasoning; the hand-rolled JSON parsing there
has the same safe-fallback property Instructor would otherwise provide.

## Repo layout

```
ally/
├── services/
│   ├── retrieval/      # Qdrant + BM25 hybrid search, reranking, MMR 
│   ├── llm_gateway/    # LiteLLM router: Qwen3 (primary) + OpenAI (fallback), circuit breaker
│   ├── chat/           # Prompt building, query condensation, MongoDB conversation history 
│   ├── auth/           # API-key auth, tenant resolution, tenant provisioning (Phase 2)
│   ├── agent/          # LangGraph graph, classifier, tool registry, connected-accounts vault 
│   ├── api/            # FastAPI app: REST + WebSocket chat + PC agent bridge endpoint
│   └── ingestion/        # upload use case and local/S3 artifact storage
├── local_agent/        # Standalone client: runs on the user's own PC, connects out via WebSocket 
├── workflows/          # Temporal workflow/worker entrypoints + activities 
├── eval/               # Ragas eval sets + scripts (Recall@K, etc.)
└── infra/
    ├── docker-compose.yml
    └── postgres/       # SQL init scripts (chunks, tenancy/auth, agent integrations)
```

`clustering/` (Phase 6) and `observability/` (Phase 7) directories will
be added when those phases start — they don't exist yet, intentionally.

## Phase roadmap

- [x] Phase 0 — Infra bootstrap (this scaffold)
- [x] Phase 1 — Ingestion & OCR pipeline (Temporal + Unstructured/Textract)
- [x] Phase 2 — Hybrid retrieval + Ragas eval, measured on a 121-document labeled corpus (Recall@5 table below)
- [x] Phase 3 — LLM gateway + streaming chat with citations
- [x] Phase 4 — LangGraph agent layer (RAG vs. tool actions vs. local PC control)
- [x] Phase 4 — LangGraph agent layer (RAG vs. tool actions vs. local PC control)
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

# Migrate a backed-up/restored-copy-rehearsed database before using the new auth schema
python -m alembic upgrade head

# 4. Provision a tenant and API key (all endpoints require one)
#    Prints the raw key once -- it is stored only as a SHA-256 hash.
python -m services.auth.provision_tenant create "My Org"

# 5. Backfill legacy document_chunks rows into Qdrant and BM25
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
- Native WebSocket clients: `X-API-Key` header.
- Browsers: obtain an authenticated single-use ticket via `POST /auth/ws-tickets`,
  then connect to `/ws/chat?ticket=...`. Production rejects query API keys.

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
`http://127.0.0.1:8002/v1`) and `QWEN_MODEL_NAME` to the model id that
server reports.

With vLLM (production, GPU):

```bash
vllm serve Qwen/Qwen3-7B --port 8002
```

With `llama.cpp` (flags depend on your hardware):

```bash
llama-server -hf Qwen/Qwen3-7B --port 8002 -c <context window> -np 1 -dev Vulkan1 -ngl 28
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
   -> user + full answer atomically persisted                        [services/chat/mongo_history.py]
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

**Conversation history** (`services/chat/mongo_history.py`): an injected,
lifespan-owned Mongo adapter stores each user/assistant exchange in one atomic
`$push`/`$each`/`$slice`, scoped by `(tenant_id, conversation_id)`. The even
`CHAT_HISTORY_MAX_MESSAGES` limit retains complete new pairs. `done` follows
acknowledged saving; an unconfirmed acknowledgement returns
`history_write_unconfirmed`, because the pair may already be stored. Tool action
outcome is separate from `history_status`; never repeat an action just to repair
history. Legacy helpers remain in `services/chat/history.py`. A WebSocket gets a
fresh `conversation_id` unless the client passes `?conversation_id=...` to resume.

**Follow-up turns queries**: Follow-up turns are condensed into 
a standalone retrieval query before hitting Qdrant/BM25 
(`services/chat/condense.py`) — e.g. "what about page 2?" is rewritten 
using prior turns before it's used for search. This only affects 
the retrieval query; the LLM prompt still receives the original message
and full history. Skipped on the first turn of a conversation, and fails
open (falls back to the raw query) if the condensation call itself fails.

## Agent layer: tools, commands, and the local PC agent (Phase 4)

Every chat message now runs through `services/agent/graph.py` before
anything else: a LangGraph `StateGraph` classifies the message into
`rag` / `tool` / `conversational` and, for `tool`, executes it. The
`rag`/`conversational` paths hand back to `chat_ws`, which streams the
answer exactly as in Phase 3 (the second case is the same code path with
an empty sources list — no document search for small talk).

```
message
  -> /tool.action key=value ...?  -- explicit command, skip straight to dispatch
  -> else: LLM classifier (services/agent/classifier.py) -> {rag, tool, conversational}
  -> "tool": ToolRegistry.get(name).execute(tenant_id, action, params)
  -> "rag" | "conversational": chat_ws streams as in Phase 3
```

### Explicit commands

`/tool_name.action key=value key2="quoted value"` always bypasses the
classifier — your own stated intent beats a model's guess, and it's the
escape hatch when the classifier gets it wrong. See `services/agent/commands.py`.

```
/jira.create_ticket summary="Fix login bug" description="Users can't log in"
/local_pc.open_app app_name=notepad
/calendar.list_events start=2026-01-01 end=2026-01-31
```

### Tools (Jira, Drive, Calendar) — mocked, vault-ready

`JiraTool`/`DriveTool`/`CalendarTool` return realistically-shaped mock
responses (no real Atlassian/Google OAuth app exists yet) but the
credential check is real: each one calls
`services.agent.connected_accounts.get_connected_account()` and refuses
to run if nothing is connected. Swapping in the real API later is a
matter of replacing the mock body of each action with an actual HTTP
call using the decrypted token — the classifier, the graph, and the chat
protocol don't change.

**Connected-accounts vault**: tokens are encrypted with Fernet
(`cryptography`) before ever reaching Postgres — the `connected_accounts`
table only stores ciphertext. Requires `CONNECTED_ACCOUNTS_ENCRYPTION_KEY`
(generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
Until a real OAuth flow exists, connect an account manually:

```python
from services.agent.connected_accounts import save_connected_account
save_connected_account("<tenant-id>", "jira", access_token="...")
```

### Adding a new tool

1. Subclass `Tool` (`services/agent/tools/base.py`), implement `execute()`.
2. Register an instance in `services/agent/registry.py`.

The classifier and the chat protocol pick it up automatically via
`ToolRegistry.describe_for_classifier()` — no other file needs to change.

### Local PC agent

`local_pc` is different from the other three tools: instead of an
outbound API call, the backend sends a command down a WebSocket the
user's own machine opened *to* the backend (`/ws/pc-agent`), since the
backend can't reach a device behind a home/office NAT directly. See
`services/agent/pc_agent_bridge.py` for the connection/correlation
registry and `local_agent/README.md` for setup.

Four independently-toggleable capabilities, each gated on the local
machine — **not** by the server, which cannot force a local machine to
do anything its own operator hasn't allowed:

| Capability | Risk |
|---|---|
| `open_apps` — launch apps, manage windows | Low |
| `file_rw` — read/write files, restricted to `allowed_folders` | Low, if scoped tightly |
| `shell_exec` — run arbitrary shell commands | **High** — no sandbox; every command is audit-logged locally |
| `mouse_keyboard_control` — move mouse, click, type, press keys | **High** — can interact with anything visible on screen |

Deliberately out of scope for v1: screenshots, clipboard access, system
info. See `local_agent/README.md` for the full safety model, including
why `shell_exec` has no command allowlist (a blocklist is not a real
security boundary and a safe-enough allowlist wouldn't be useful).

### Pluggable classifier

`INTENT_CLASSIFIER` (default `llm`) selects the implementation in
`services/agent/classifier.py`'s `_CLASSIFIERS` registry. A future
custom-trained classifier ("JEV") is one new entry plus an env var
change — the graph and the rest of the chat flow don't need to know
which implementation is making the routing decision.

### Known limitations

- The classifier sees only the current message, not conversation
  history — a follow-up like "do that again" won't resolve to a prior
  tool call. Same category of limitation query condensation addressed
  for RAG retrieval; not yet addressed here.
- `PCAgentBridge`'s connection registry is in-process — works for a
  single backend instance (where this project actually is today), would
  need moving to shared state (e.g. Redis) to support horizontal scaling.
- Only one PC agent connection is kept per tenant; a second one replaces
  the first rather than supporting multiple paired devices.

## Agent layer: tools, commands, and the local PC agent (Phase 4)

Every chat message now runs through `services/agent/graph.py` before
anything else: a LangGraph `StateGraph` classifies the message into
`rag` / `tool` / `conversational` and, for `tool`, executes it. The
`rag`/`conversational` paths hand back to `chat_ws`, which streams the
answer exactly as in Phase 3 (the second case is the same code path with
an empty sources list — no document search for small talk).

```
message
  -> /tool.action key=value ...?  -- explicit command, skip straight to dispatch
  -> else: LLM classifier (services/agent/classifier.py) -> {rag, tool, conversational}
  -> "tool": ToolRegistry.get(name).execute(tenant_id, action, params)
  -> "rag" | "conversational": chat_ws streams as in Phase 3
```

### Explicit commands

`/tool_name.action key=value key2="quoted value"` always bypasses the
classifier — your own stated intent beats a model's guess, and it's the
escape hatch when the classifier gets it wrong. See `services/agent/commands.py`.

```
/jira.create_ticket summary="Fix login bug" description="Users can't log in"
/local_pc.open_app app_name=notepad
/calendar.list_events start=2026-01-01 end=2026-01-31
```

### Tools (Jira, Drive, Calendar) — mocked, vault-ready

`JiraTool`/`DriveTool`/`CalendarTool` return realistically-shaped mock
responses (no real Atlassian/Google OAuth app exists yet) but the
credential check is real: each one calls
`services.agent.connected_accounts.get_connected_account()` and refuses
to run if nothing is connected. Swapping in the real API later is a
matter of replacing the mock body of each action with an actual HTTP
call using the decrypted token — the classifier, the graph, and the chat
protocol don't change.

**Connected-accounts vault**: tokens are encrypted with Fernet
(`cryptography`) before ever reaching Postgres — the `connected_accounts`
table only stores ciphertext. Requires `CONNECTED_ACCOUNTS_ENCRYPTION_KEY`
(generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
Until a real OAuth flow exists, connect an account manually:

```python
from services.agent.connected_accounts import save_connected_account
save_connected_account("<tenant-id>", "jira", access_token="...")
```

### Adding a new tool

1. Subclass `Tool` (`services/agent/tools/base.py`), implement `execute()`.
2. Register an instance in `services/agent/registry.py`.

The classifier and the chat protocol pick it up automatically via
`ToolRegistry.describe_for_classifier()` — no other file needs to change.

### Local PC agent

`local_pc` is different from the other three tools: instead of an
outbound API call, the backend sends a command down a WebSocket the
user's own machine opened *to* the backend (`/ws/pc-agent`), since the
backend can't reach a device behind a home/office NAT directly. See
`services/agent/pc_agent_bridge.py` for the connection/correlation
registry and `local_agent/README.md` for setup.

Four independently-toggleable capabilities, each gated on the local
machine — **not** by the server, which cannot force a local machine to
do anything its own operator hasn't allowed:

| Capability | Risk |
|---|---|
| `open_apps` — launch apps, manage windows | Low |
| `file_rw` — read/write files, restricted to `allowed_folders` | Low, if scoped tightly |
| `shell_exec` — run arbitrary shell commands | **High** — no sandbox; every command is audit-logged locally |
| `mouse_keyboard_control` — move mouse, click, type, press keys | **High** — can interact with anything visible on screen |

Deliberately out of scope for v1: screenshots, clipboard access, system
info. See `local_agent/README.md` for the full safety model, including
why `shell_exec` has no command allowlist (a blocklist is not a real
security boundary and a safe-enough allowlist wouldn't be useful).

### Pluggable classifier

`INTENT_CLASSIFIER` (default `llm`) selects the implementation in
`services/agent/classifier.py`'s `_CLASSIFIERS` registry. A future
custom-trained classifier ("JEV") is one new entry plus an env var
change — the graph and the rest of the chat flow don't need to know
which implementation is making the routing decision.

### Known limitations

- The classifier sees only the current message, not conversation
  history — a follow-up like "do that again" won't resolve to a prior
  tool call. Same category of limitation query condensation addressed
  for RAG retrieval; not yet addressed here.
- `PCAgentBridge`'s connection registry is in-process — works for a
  single backend instance (where this project actually is today), would
  need moving to shared state (e.g. Redis) to support horizontal scaling.
- Only one PC agent connection is kept per tenant; a second one replaces
  the first rather than supporting multiple paired devices.

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

Configuration is validated as an immutable process snapshot. Set variables before
starting each process and restart to apply changes.
- `PROJECT_NAME` (defaults to `Ally`)
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `EMBEDDING_MODEL`, `RERANKER_MODEL`
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
