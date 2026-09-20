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


# Matches the "default" tenant seeded by
# infra/postgres/002_auth_and_tenancy.sql. Retrieval is tenant-scoped, so
# the eval set must be evaluated against whichever tenant actually owns
# the ingested corpus — override with --tenant-id when that isn't the
# default tenant.
DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000000"


# Each config below changes exactly one stage relative to the previous
# one, so a recall delta between two adjacent rows can be attributed to
# that single stage. The earlier version of this table jumped straight
# from "hybrid" to a config that added BOTH reranking and MMR at once —
# any observed improvement (or regression) was impossible to attribute to
# either stage individually.
PIPELINE_CONFIGS: dict[str, PipelineConfig] = {
    "vector_only": {"use_bm25": False, "use_rerank": False, "use_mmr": False},
    "hybrid": {"use_bm25": True, "use_rerank": False, "use_mmr": False},
    "hybrid_reranked": {"use_bm25": True, "use_rerank": True, "use_mmr": False},
}

# MMR is evaluated separately from the recall comparison above, and is
# NOT part of PIPELINE_CONFIGS. MMR trades relevance for diversity by
# design — penalizing near-duplicate chunks even when they're relevant —
# so it is expected to hold recall flat or reduce it. Reporting it in the
# same recall table as the other stages invites the wrong conclusion
# ("MMR made retrieval worse"), when what it actually did is what it's
# supposed to do: trade a small amount of relevance for less redundant
# context. Its effect belongs in an answer-quality/diversity eval, not a
# recall benchmark.
MMR_CONFIG: PipelineConfig = {"use_bm25": True, "use_rerank": True, "use_mmr": True}


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


def _evaluate_config(
    eval_queries: list[dict],
    config_name: str,
    config_kwargs: PipelineConfig,
    k: int,
    tenant_id: str,
) -> tuple[float, list[dict]]:
    """Runs one pipeline config over every query and returns its average
    Recall@K plus per-query rows for the results file.

    ``tenant_id`` scopes every retrieval to the tenant whose corpus the
    eval set was labeled against — retrieval is tenant-scoped by design
    (see services.retrieval.pipeline.retrieve), so evaluating against the
    wrong tenant returns nothing and scores 0.0 across the board.
    """
    scores: list[float] = []
    per_query: list[dict] = []

    for entry in eval_queries:
        retrieved = retrieve(entry["query"], tenant_id, top_k=k, **config_kwargs)
        score = recall_at_k(retrieved, entry["relevant_chunks"], k)
        scores.append(score)
        per_query.append({"query": entry["query"], "config": config_name, f"recall_at_{k}": score})
        log.info(
            "query_evaluated",
            config=config_name,
            query=entry["query"][:60],
            recall_at_k=round(score, 3),
        )

    average = sum(scores) / len(scores) if scores else 0.0
    return average, per_query


def run_recall_comparison(
    eval_queries: list[dict], k: int = 5, tenant_id: str = DEFAULT_TENANT_ID
) -> tuple[dict[str, float], list[dict]]:
    """Evaluates the variable-isolated configs in PIPELINE_CONFIGS (each
    changes exactly one stage relative to the previous row) and returns
    aggregate + per-query recall. MMR is intentionally excluded — see
    run_mmr_diversity_check()."""
    averages: dict[str, float] = {}
    per_query: list[dict] = []

    for config_name, config_kwargs in PIPELINE_CONFIGS.items():
        average, rows = _evaluate_config(eval_queries, config_name, config_kwargs, k, tenant_id)
        averages[config_name] = average
        per_query.extend(rows)

    return averages, per_query


def run_mmr_diversity_check(
    eval_queries: list[dict], k: int = 5, tenant_id: str = DEFAULT_TENANT_ID
) -> tuple[float, list[dict]]:
    """Evaluates the full pipeline including MMR, reported separately from
    the recall table on purpose (see the MMR_CONFIG comment above)."""
    return _evaluate_config(eval_queries, "hybrid_reranked_mmr", MMR_CONFIG, k, tenant_id)


def run_ragas_metrics(
    eval_queries: list[dict], k: int = 5, tenant_id: str = DEFAULT_TENANT_ID
) -> object | None:
    """Evaluate hybrid-reranked contexts with Ragas when ``OPENAI_API_KEY`` is set.

    Targets the ragas >=0.2 dataset API (SingleTurnSample / EvaluationDataset),
    which is what ragas==0.4.3 (pinned in requirements.txt) uses — the
    previous version of this function called the pre-0.2
    Dataset.from_list()/evaluate(dataset, metrics=[...]) pattern, which
    that pin doesn't support at all (context_precision/context_recall
    aren't even importable as bare names anymore; they're classes that
    need instantiating, and evaluate() now requires an explicit
    evaluator LLM rather than picking one up implicitly).

    Returns ``None`` without running retrieval when the key is absent.
    """
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("ragas_skipped", reason="no LLM API key configured")
        return None

    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

    samples = []
    for entry in eval_queries:
        retrieved = retrieve(
            entry["query"], tenant_id, top_k=k, **PIPELINE_CONFIGS["hybrid_reranked"]
        )
        samples.append(
            SingleTurnSample(
                user_input=entry["query"],
                retrieved_contexts=[r["text"] for r in retrieved],
                reference=entry["ground_truth"],
            )
        )

    dataset = EvaluationDataset(samples=samples)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=evaluator_llm,
    )
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
    parser.add_argument(
        "--tenant-id",
        type=str,
        default=DEFAULT_TENANT_ID,
        help=(
            "Tenant whose corpus the eval set was labeled against. Retrieval "
            "is tenant-scoped, so pointing this at the wrong tenant returns "
            "no results and scores 0.0 across every config."
        ),
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"{eval_path} not found — copy eval/eval_set.sample.json to "
            f"{eval_path} and fill in real labeled queries against your corpus"
        )

    eval_data = json.loads(eval_path.read_text())
    eval_queries = eval_data["queries"]

    print(f"\nRunning Recall@{args.k} comparison across {len(eval_queries)} queries...")
    print("(each config below changes exactly one stage vs. the row above it)\n")
    recall_results, per_query = run_recall_comparison(
        eval_queries, k=args.k, tenant_id=args.tenant_id
    )

    print("Recall@{} results:".format(args.k))
    print("-" * 40)
    for config_name, avg_recall in recall_results.items():
        print(f"  {config_name:20s} {avg_recall:.3f}")
    print("-" * 40)

    print("\nRunning MMR diversity check (reported separately — MMR trades")
    print("relevance for diversity by design, so it is NOT expected to")
    print("improve Recall@K and should not be compared against the table above):")
    mmr_recall, mmr_per_query = run_mmr_diversity_check(
        eval_queries, k=args.k, tenant_id=args.tenant_id
    )
    print(f"  {'hybrid_reranked_mmr':20s} {mmr_recall:.3f}")

    output = {
        "num_queries": len(eval_queries),
        "tenant_id": args.tenant_id,
        "k": args.k,
        "recall_at_k": recall_results,
        "per_query": per_query,
        "mmr_diversity_check": {
            "note": (
                "MMR trades relevance for diversity by design and is evaluated "
                "separately from recall_at_k above; a lower score here than "
                "hybrid_reranked is expected, not a regression."
            ),
            "recall_at_k": mmr_recall,
            "per_query": mmr_per_query,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to {output_path}")

    print("\nRunning Ragas context_precision / context_recall on hybrid_reranked config...")
    ragas_result = run_ragas_metrics(eval_queries, k=args.k, tenant_id=args.tenant_id)
    if ragas_result is not None:
        print(ragas_result)
    else:
        print("Skipped (no LLM API key configured).")


if __name__ == "__main__":
    main()
