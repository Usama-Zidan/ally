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
- `backfill_index.py` — rebuilds both indexes from Phase 1 Postgres chunks

The Temporal ingestion workflow calls `embed_and_index` after Postgres
persistence, so new documents are indexed automatically. Run the backfill
script when migrating existing Phase 1 data:

```bash
python services/retrieval/backfill_index.py
```

The `document_chunks` schema is versioned in
`infra/postgres/001_document_chunks.sql` and is mounted into the Postgres
container during first initialization. Existing volumes must be migrated
explicitly before running ingestion.

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
