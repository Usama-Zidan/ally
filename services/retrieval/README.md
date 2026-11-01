# Retrieval service (Phase 2)

Phase 2 intentionally uses two complementary indexes:

```
query -> Qdrant dense search + BM25 lexical search
      -> reciprocal-rank fusion
      -> optional cross-encoder reranking
      -> optional MMR diversity selection
      -> top-K chunks
```

- `embeddings.py` — bge-base-en-v1.5 dense embeddings
- `qdrant_store.py` — dense Qdrant collection and deterministic chunk IDs
- `bm25_index.py` — persisted in-process BM25 lexical index
- `fusion.py` — reciprocal rank fusion
- `reranker.py` — bge-reranker-v2 cross-encoder
- `mmr.py` — diversity-aware final selection
- `pipeline.py` — retrieval entrypoint with stage toggles and metadata filters
- `backfill_index.py` — rebuilds both indexes from the legacy Postgres `document_chunks` table

The Temporal ingestion workflow stages version-specific chunks and vectors,
activates the version, then rebuilds the active BM25 view. Run the backfill
script only to load legacy `document_chunks` rows that have not been migrated:

```bash
python services/retrieval/backfill_index.py
```

The active document/version schema is managed by Alembic migrations under
`migrations/versions/`. The SQL file in `infra/postgres/` is the legacy bootstrap
schema for new development databases; existing volumes must be upgraded with
`python -m alembic upgrade head` before running the current ingestion workflow.

## Evaluation

Create `eval/eval_set.json` from the sample and hand-label at least 50 queries
against the ingested corpus. Each query needs a `ground_truth` answer and
`relevant_chunks`; labels may be exact chunk IDs or `filename:p<page>` values.

```bash
python eval/run_ragas_eval.py --eval-set eval/eval_set.json --k 5 --output eval/results.json
```

The script writes per-query and aggregate Recall@K results for vector-only,
hybrid, and hybrid+reranked configurations. Ragas context metrics run when an
LLM provider key is configured.
