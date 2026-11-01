"""Measure configured retrieval-model cold loads and warm inference locally.

The benchmark runs with Hugging Face Hub offline so it cannot silently download
weights. Example::

    python scripts/benchmark_retrieval_models.py --iterations 10 --warmup 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import statistics
import sys
import time
from typing import Callable
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import runtime_settings  # noqa: E402
from services.retrieval.model_loading import configure_model_inference  # noqa: E402
from services.retrieval.model_runtime import RetrievalModels  # noqa: E402


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _measure(function: Callable[[], object], warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    total = sum(samples)
    return {
        "iterations": iterations,
        "p50_seconds": _percentile(samples, 0.50),
        "p95_seconds": _percentile(samples, 0.95),
        "mean_seconds": statistics.fmean(samples),
        "requests_per_second": iterations / total if total else 0.0,
    }


def _hardware() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "gpu": None,
        "gpu_memory_total_bytes": None,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
        info["nvidia_smi_gpus"] = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
    except (OSError, subprocess.SubprocessError):
        info["nvidia_smi_gpus"] = []
    try:
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            info["gpu"] = device.name
            info["gpu_memory_total_bytes"] = device.total_memory
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch_version"] = None
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.candidates < 1:
        parser.error("iterations/candidates must be positive and warmup must be nonnegative")

    settings = runtime_settings.application
    configure_model_inference(settings)
    models = RetrievalModels(settings.embedding_model, settings.reranker_model)
    query = "What is the annual leave policy?"
    candidates = [
        {
            "text": f"Policy passage {index}: employees may request annual leave.",
            "filename": "policy.txt",
        }
        for index in range(args.candidates)
    ]

    embedding_load_started = time.perf_counter()
    models.embed_query(query)
    embedding_load_seconds = time.perf_counter() - embedding_load_started
    reranker_load_started = time.perf_counter()
    models.rerank(query, candidates.copy(), top_k=min(8, args.candidates))
    reranker_load_seconds = time.perf_counter() - reranker_load_started

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None  # type: ignore[assignment]

    embedding = _measure(lambda: models.embed_query(query), args.warmup, args.iterations)
    reranking = _measure(
        lambda: models.rerank(query, candidates.copy(), top_k=min(8, args.candidates)),
        args.warmup,
        args.iterations,
    )
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_memory_allocated_bytes = torch.cuda.memory_allocated()
        gpu_memory_peak_bytes = torch.cuda.max_memory_allocated()
    else:
        gpu_memory_allocated_bytes = None
        gpu_memory_peak_bytes = None

    report = {
        "hardware": _hardware(),
        "models": {
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
        },
        "offline_only": os.environ.get("HF_HUB_OFFLINE") == "1",
        "candidate_count": args.candidates,
        "process_first_request_seconds": {
            "embedding_first_query": embedding_load_seconds,
            "reranker_first_call": reranker_load_seconds,
        },
        "warm_inference": {"embedding_query": embedding, "rerank_candidates": reranking},
        "gpu_memory_allocated_bytes": gpu_memory_allocated_bytes,
        "gpu_memory_peak_bytes": gpu_memory_peak_bytes,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
