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
│   ├── agent/         # LangGraph graph: intent -> RAG node | tool node
│   ├── llm-gateway/   # LiteLLM proxy config (Qwen3 + fallback providers)
│   └── api/           # FastAPI app: REST + WebSocket chat
├── workflows/         # Temporal workflow/worker entrypoints
├── eval/              # Ragas eval sets + scripts (Recall@K, etc.)
├── clustering/         # BERTopic pipeline
├── infra/
│   └── docker-compose.yml
└── observability/      # Langfuse/OTel config
```

## Phase roadmap

- [x] Phase 0 — Infra bootstrap (this scaffold)
- [x] Phase 1 — Ingestion & OCR pipeline (Temporal + Unstructured/Textract)
- [ ] Phase 2 — Hybrid retrieval + Ragas eval
- [ ] Phase 3 — LLM gateway + streaming chat with citations
- [ ] Phase 4 — LangGraph agent layer (RAG vs. tool actions)
- [ ] Phase 5 — Semantic + exact-match caching
- [ ] Phase 6 — BERTopic clustering
- [ ] Phase 7 — Observability & hardening

## Getting started

```bash
# 1. Bring up infra
cd infra
docker compose up -d

# 2. Verify services
#    Qdrant:      http://localhost:6333/dashboard
#    Temporal UI: http://localhost:8080
#    Langfuse:    http://localhost:3001
#    RedisInsight: http://localhost:8001

# 3. Install Python deps (in a venv)
cd ..
pip install -r requirements.txt

# 4. Run the API
uvicorn services.api.main:app --reload --port 8000

# 5. Run the ingestion worker (separate terminal)
python workflows/worker.py
```

## Environment variables

Copy `.env.example` to `.env` and fill in:
- `PROJECT_NAME` (defaults to `Ally`)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (for Textract)
- `OPENAI_API_KEY`, `COHERE_API_KEY`, `DEEPSEEK_API_KEY` (LiteLLM fallback providers)
- `QDRANT_URL`, `POSTGRES_DSN`, `MONGO_URI`, `REDIS_URL`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
