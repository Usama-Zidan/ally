"""Bounded process-wide inference admission behavior."""

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import Mock, patch

from services.retrieval import embeddings, reranker
from services.retrieval.model_loading import (
    ModelInferenceGate,
    ModelInferenceRejected,
)


class ModelInferenceGateTest(unittest.TestCase):
    def test_limits_active_calls_and_bounds_waiting_callers(self) -> None:
        gate = ModelInferenceGate(concurrency=1, max_waiters=1, wait_timeout_seconds=1)
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        def hold_first() -> None:
            with gate.inference():
                first_entered.set()
                release_first.wait(timeout=2)

        def wait_second() -> None:
            with gate.inference():
                second_entered.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(hold_first)
            self.assertTrue(first_entered.wait(timeout=1))
            second = executor.submit(wait_second)
            deadline = time.monotonic() + 1
            while gate.snapshot()["waiting"] != 1 and time.monotonic() < deadline:
                time.sleep(0.005)

            self.assertEqual(gate.snapshot(), {"active": 1, "waiting": 1, "rejected": 0})
            with self.assertRaises(ModelInferenceRejected):
                with gate.inference():
                    self.fail("a third caller must not exceed active plus waiting capacity")
            release_first.set()
            first.result(timeout=1)
            second.result(timeout=1)

        self.assertTrue(second_entered.is_set())
        self.assertEqual(gate.snapshot(), {"active": 0, "waiting": 0, "rejected": 1})

    def test_wait_timeout_rejects_without_leaking_queue_capacity(self) -> None:
        gate = ModelInferenceGate(concurrency=1, max_waiters=1, wait_timeout_seconds=0.01)
        with gate.inference():
            with self.assertRaisesRegex(ModelInferenceRejected, "timed out"):
                with gate.inference():
                    self.fail("the second caller should time out")
            self.assertEqual(gate.snapshot()["waiting"], 0)
        self.assertEqual(gate.snapshot()["rejected"], 1)

    def test_embedding_and_reranking_calls_use_the_gate(self) -> None:
        calls: list[str] = []

        @contextmanager
        def admission():
            calls.append("enter")
            yield

        gate = Mock(inference=admission)
        encoder = Mock(encode=Mock(return_value=Mock(tolist=Mock(return_value=[0.1]))))
        cross_encoder = Mock(predict=Mock(return_value=[0.9]))
        with (
            patch.object(embeddings, "MODEL_INFERENCE_GATE", gate),
            patch.object(embeddings, "_get_model", return_value=encoder),
            patch.object(reranker, "MODEL_INFERENCE_GATE", gate),
            patch.object(reranker, "_get_reranker", return_value=cross_encoder),
        ):
            self.assertEqual(embeddings.embed_query("question"), [0.1])
            ranked = reranker.rerank(
                "question", [{"text": "answer", "filename": "source.txt"}], top_k=1
            )

        self.assertEqual(calls, ["enter", "enter"])
        self.assertEqual(ranked[0]["rerank_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
