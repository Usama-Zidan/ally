"""
Periodic flusher for the BM25 index.

rebuild_index_from_postgres() debounces full rebuilds to at most one per
BM25_REBUILD_MIN_INTERVAL_SECONDS (see bm25_index.py) — a burst of
document ingestions no longer means a burst of full-corpus rebuilds. But
that debounce means the *last* document in a burst can leave the index
"dirty" (chunks in Postgres that aren't in the BM25 index yet) with
nothing scheduled to catch it up if no further ingestion happens to land
outside the debounce window.

This script closes that gap: run it as a cron job, a Temporal Schedule,
or a long-running sidecar container, and it will perform exactly one
rebuild whenever (and only when) the index is actually dirty.

Usage:
    python -m services.retrieval.bm25_rebuild_worker --once
    python -m services.retrieval.bm25_rebuild_worker              # loop
"""
from __future__ import annotations

import argparse
import time

import structlog

from services.retrieval.bm25_index import flush_if_dirty

log = structlog.get_logger()


def run_once() -> None:
    index = flush_if_dirty()
    if index is not None:
        log.info("bm25_index_flushed", num_documents=len(index.doc_ids))
    else:
        log.debug("bm25_index_not_dirty_skipping")


def run_loop(poll_interval_seconds: float = 30.0) -> None:
    log.info("bm25_rebuild_worker_started", poll_interval_seconds=poll_interval_seconds)
    while True:
        run_once()
        time.sleep(poll_interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single flush check and exit (for cron/Temporal Schedule use).",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=30.0,
        help="How often to check for dirty state in loop mode.",
    )
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_loop(args.poll_interval_seconds)


if __name__ == "__main__":
    main()
