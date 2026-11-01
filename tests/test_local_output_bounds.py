"""Local file/list results stay bounded before transport serialization."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

LOCAL = str(Path(__file__).resolve().parents[1] / "local_agent")
if LOCAL not in sys.path:
    sys.path.insert(0, LOCAL)
import agent  # noqa: E402


class LocalOutputBoundsTest(unittest.TestCase):
    def test_large_unicode_file_is_explicitly_truncated_and_fits_socket(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.txt"
            path.write_text("\U0001f600" * (agent.MAX_FILE_CHARS * 10), encoding="utf-8")
            local = agent.LocalAgent(
                {"capabilities": {"file_rw": True}, "allowed_folders": [temporary]},
                approve=lambda action, params: True,
            )
            result = local.read_file({"path": str(path)})
            self.assertEqual(len(result["content"]), agent.MAX_FILE_CHARS)
            self.assertTrue(result["truncated"])
            encoded = agent._encode_result("request", result)
            self.assertLessEqual(len(encoded.encode("utf-8")), agent.MAX_RESULT_BYTES)
            self.assertTrue(json.loads(encoded)["details"]["truncated"])

    def test_listing_limits_count_and_characters_without_consuming_remaining_iterator(self):
        consumed = []

        def names():
            for i in range(100000):
                consumed.append(i)
                yield str(i)

        result, truncated = agent._bounded_names(names())
        self.assertTrue(truncated)
        self.assertEqual(len(result), agent.MAX_LIST_ENTRIES)
        self.assertEqual(len(consumed), agent.MAX_LIST_ENTRIES + 1)
        result, truncated = agent._bounded_names(["x" * agent.MAX_LIST_CHARS, "another"])
        self.assertTrue(truncated)
        self.assertEqual(len(result), 1)

    def test_small_directory_is_complete_and_large_directory_is_marked_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            local = agent.LocalAgent(
                {"capabilities": {"file_rw": True}, "allowed_folders": [temporary]},
                approve=lambda action, params: True,
            )
            path = Path(temporary)
            (path / "small").touch()
            self.assertFalse(local.list_directory({"path": temporary})["truncated"])
            for i in range(agent.MAX_LIST_ENTRIES + 1):
                (path / str(i)).touch()
            result = local.list_directory({"path": temporary})
            self.assertTrue(result["truncated"])
            self.assertEqual(len(result["entries"]), agent.MAX_LIST_ENTRIES)

    def test_any_remaining_oversized_action_result_is_rejected_before_send(self):
        with self.assertRaises(ValueError):
            agent._encode_result("request", {"unexpected": "x" * agent.MAX_RESULT_BYTES})


if __name__ == "__main__":
    unittest.main()
