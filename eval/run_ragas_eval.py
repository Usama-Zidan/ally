"""
Retrieval evaluation script.

Two layers of evaluation, run against the same eval_set.json:

1. Recall@K (manual, no LLM needed) — for each query, checks whether the
   hand-labeled relevant chunks appear anywhere in the top-K retrieved
   results. Run once per pipeline config (vector-only / hybrid /
   hybrid+rerank) to produce the comparison table that justifies the
   hybrid approach.

2. Ragas metrics (context_precision, context_recall) — LLM-judged metrics that need a
   ground-truth answer,
   not just retrieved chunks. Requires an LLM configured via LiteLLM/
   OpenAI; skipped gracefully if no API key is set so this script still
   runs end-to-end without Phase 3's LLM gateway in place.

Usage:
    python eval/run_ragas_eval.py --eval-set eval/eval_set.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import structlog

from services.retrieval.pipeline import retrieve

log = structlog.get_logger()


class PipelineConfig(TypedDict):
    use_bm25: bool
    use_rerank: bool
    use_mmr: bool


PIPELINE_CONFIGS: dict[str, PipelineConfig] = {
    "vector_only": {"use_bm25": False, "use_rerank": False, "use_mmr": False},
    "hybrid": {"use_bm25": True, "use_rerank": False, "use_mmr": False},
    "hybrid_reranked": {"use_bm25": True, "use_rerank": True, "use_mmr": True},
}


def chunk_keys(result: dict) -> set[str]:
    """Support exact chunk labels and the sample's page-level labels."""
    return {
        result["id"],
        f"{result['filename']}:p{result['page_number']}",
    }


def recall_at_k(retrieved: list[dict], relevant_chunks: list[str], k: int) -> float:
    """Return the fraction of relevant chunk or page labels found in the first ``k`` results."""
    top_k_keys = set().union(*(chunk_keys(r) for r in retrieved[:k]))
    relevant_set = set(relevant_chunks)
    if not relevant_set:
        return 0.0
    hits = len(top_k_keys & relevant_set)
    return hits / len(relevant_set)


def run_recall_comparison(
    eval_queries: list[dict], k: int = 5
) -> tuple[dict[str, float], list[dict]]:
    """Evaluate each pipeline configuration and return aggregate and per-query recall."""
    results: dict[str, list[float]] = {name: [] for name in PIPELINE_CONFIGS}
    per_query: list[dict] = []

    for entry in eval_queries:
        query = entry["query"]
        relevant_chunks = entry["relevant_chunks"]

        for config_name, config_kwargs in PIPELINE_CONFIGS.items():
            retrieved = retrieve(query, top_k=k, **config_kwargs)
            score = recall_at_k(retrieved, relevant_chunks, k)
            results[config_name].append(score)
            per_query.append(
                {"query": query, "config": config_name, f"recall_at_{k}": score}
            )
            log.info(
                "query_evaluated",
                config=config_name,
                query=query[:60],
                recall_at_k=round(score, 3),
            )

    averages = {
        name: sum(scores) / len(scores) if scores else 0.0
        for name, scores in results.items()
    }
    return averages, per_query


def run_ragas_metrics(eval_queries: list[dict], k: int = 5) -> object | None:
    """Evaluate hybrid-reranked contexts with Ragas when ``OPENAI_API_KEY`` is set.

    Returns ``None`` without running retrieval when the key is absent.
    """
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("ragas_skipped", reason="no LLM API key configured")
        return None

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall

    rows = []
    for entry in eval_queries:
        retrieved = retrieve(entry["query"], top_k=k, **PIPELINE_CONFIGS["hybrid_reranked"])
        rows.append(
            {
                "question": entry["query"],
                "contexts": [r["text"] for r in retrieved],
                "ground_truth": entry["ground_truth"],
            }
        )

    dataset = Dataset.from_list(rows)
    result = evaluate(dataset, metrics=[context_precision, context_recall])
    return result


def main() -> None:
    """Run retrieval evaluation and write the recall comparison as JSON.

    Raises:
        FileNotFoundError: If the requested evaluation set does not exist.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=str, default="eval/eval_set.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=str, default="eval/results.json")
    args = parser.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"{eval_path} not found — copy eval/eval_set.sample.json to "
            f"{eval_path} and fill in real labeled queries against your corpus"
        )

    eval_data = json.loads(eval_path.read_text())
    eval_queries = eval_data["queries"]

    print(f"\nRunning Recall@{args.k} comparison across {len(eval_queries)} queries...\n")
    recall_results, per_query = run_recall_comparison(eval_queries, k=args.k)

    print("Recall@{} results:".format(args.k))
    print("-" * 40)
    output = {
        "num_queries": len(eval_queries),
        "k": args.k,
        "recall_at_k": recall_results,
        "per_query": per_query,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved recall comparison to {output_path}")
    for config_name, avg_recall in recall_results.items():
        print(f"  {config_name:20s} {avg_recall:.3f}")
    print("-" * 40)

    print("\nRunning Ragas context_precision / context_recall on hybrid_reranked config...")
    ragas_result = run_ragas_metrics(eval_queries, k=args.k)
    if ragas_result is not None:
        print(ragas_result)
    else:
        print("Skipped (no LLM API key configured).")


if __name__ == "__main__":
    main()
