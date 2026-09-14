import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from eval import run_ragas_eval as evaluation


class RecallEvaluationTest(unittest.TestCase):
    def test_chunk_keys_supports_exact_and_page_level_labels(self):
        self.assertEqual(
            evaluation.chunk_keys(
                {"id": "chunk-id", "filename": "policy.pdf", "page_number": 4}
            ),
            {"chunk-id", "policy.pdf:p4"},
        )

    def test_recall_at_k_counts_unique_relevant_hits_within_boundary(self):
        retrieved = [
            {"id": "one", "filename": "a.pdf", "page_number": 1},
            {"id": "two", "filename": "b.pdf", "page_number": 2},
        ]

        self.assertEqual(evaluation.recall_at_k(retrieved, ["one", "b.pdf:p2"], 1), 0.5)
        self.assertEqual(evaluation.recall_at_k(retrieved, ["one", "b.pdf:p2"], 2), 1.0)
        self.assertEqual(evaluation.recall_at_k(retrieved, [], 2), 0.0)

    def test_run_recall_comparison_evaluates_every_query_and_configuration(self):
        queries = [
            {"query": "first", "relevant_chunks": ["first-id"]},
            {"query": "second", "relevant_chunks": ["second-id"]},
        ]

        def retrieve(query, top_k, **configuration):
            result_id = f"{query}-id" if configuration["use_bm25"] else "miss"
            return [{"id": result_id, "filename": "a.pdf", "page_number": 1}]

        with patch.object(evaluation, "retrieve", side_effect=retrieve) as retrieve_mock:
            averages, per_query = evaluation.run_recall_comparison(queries, k=3)

        self.assertEqual(
            averages,
            {"vector_only": 0.0, "hybrid": 1.0, "hybrid_reranked": 1.0},
        )
        self.assertEqual(len(per_query), len(queries) * len(evaluation.PIPELINE_CONFIGS))
        self.assertEqual(retrieve_mock.call_count, 6)
        for invocation in retrieve_mock.call_args_list:
            self.assertEqual(invocation.kwargs["top_k"], 3)

    def test_run_recall_comparison_handles_empty_evaluation_set(self):
        averages, per_query = evaluation.run_recall_comparison([])

        self.assertEqual(averages, {name: 0.0 for name in evaluation.PIPELINE_CONFIGS})
        self.assertEqual(per_query, [])


class RagasEvaluationTest(unittest.TestCase):
    def test_ragas_is_skipped_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(evaluation.run_ragas_metrics([{"query": "unused"}]))

    def test_ragas_builds_dataset_from_retrieved_contexts(self):
        dataset_type = Mock()
        dataset = object()
        dataset_type.from_list.return_value = dataset
        evaluate = Mock(return_value={"context_recall": 0.75})
        context_precision = object()
        context_recall = object()

        datasets_module = types.ModuleType("datasets")
        datasets_module.Dataset = dataset_type
        ragas_module = types.ModuleType("ragas")
        ragas_module.evaluate = evaluate
        metrics_module = types.ModuleType("ragas.metrics")
        metrics_module.context_precision = context_precision
        metrics_module.context_recall = context_recall

        entry = {
            "query": "What is the leave policy?",
            "ground_truth": "Employees receive annual leave.",
        }
        retrieved = [{"text": "Annual leave is available."}]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch.dict(
            sys.modules,
            {
                "datasets": datasets_module,
                "ragas": ragas_module,
                "ragas.metrics": metrics_module,
            },
        ), patch.object(evaluation, "retrieve", return_value=retrieved) as retrieve:
            result = evaluation.run_ragas_metrics([entry], k=2)

        retrieve.assert_called_once_with(
            entry["query"],
            top_k=2,
            **evaluation.PIPELINE_CONFIGS["hybrid_reranked"],
        )
        dataset_type.from_list.assert_called_once_with(
            [
                {
                    "question": entry["query"],
                    "contexts": ["Annual leave is available."],
                    "ground_truth": entry["ground_truth"],
                }
            ]
        )
        evaluate.assert_called_once_with(
            dataset, metrics=[context_precision, context_recall]
        )
        self.assertEqual(result, {"context_recall": 0.75})


class EvaluationCommandTest(unittest.TestCase):
    def test_main_writes_recall_results_to_requested_output(self):
        with TemporaryDirectory() as directory:
            eval_path = Path(directory, "eval.json")
            output_path = Path(directory, "results", "recall.json")
            eval_path.write_text(
                json.dumps(
                    {
                        "queries": [
                            {
                                "query": "leave",
                                "relevant_chunks": ["one"],
                                "ground_truth": "answer",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            averages = {name: 0.5 for name in evaluation.PIPELINE_CONFIGS}
            details = [{"query": "leave", "config": "hybrid", "recall_at_3": 0.5}]

            with patch.object(
                sys,
                "argv",
                [
                    "run_ragas_eval.py",
                    "--eval-set",
                    str(eval_path),
                    "--output",
                    str(output_path),
                    "--k",
                    "3",
                ],
            ), patch.object(
                evaluation, "run_recall_comparison", return_value=(averages, details)
            ) as recall, patch.object(
                evaluation, "run_ragas_metrics", return_value=None
            ) as ragas:
                evaluation.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["num_queries"], 1)
        self.assertEqual(payload["k"], 3)
        self.assertEqual(payload["recall_at_k"], averages)
        self.assertEqual(payload["per_query"], details)
        recall.assert_called_once()
        ragas.assert_called_once()

    def test_main_rejects_missing_evaluation_file(self):
        with TemporaryDirectory() as directory, patch.object(
            sys, "argv", ["run_ragas_eval.py", "--eval-set", str(Path(directory, "missing.json"))]
        ):
            with self.assertRaisesRegex(FileNotFoundError, "copy eval/eval_set.sample.json"):
                evaluation.main()


if __name__ == "__main__":
    unittest.main()
