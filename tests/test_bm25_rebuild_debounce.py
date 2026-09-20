"""Tests for the debounced BM25 rebuild path.

Rebuilding is O(corpus size), so embed_and_index must not trigger a full
rebuild on every single ingested document. These tests pin the debounce
contract: skip-and-mark-dirty inside the window, rebuild outside it, and
a flusher that converges whatever the debounce skipped.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from services.retrieval import bm25_index


class Bm25RebuildDebounceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        base = Path(self._tmp.name)
        # Redirect every on-disk artifact the debounce bookkeeping touches
        # so tests never read or write a developer's real index.
        self._patches = [
            patch.object(bm25_index, "INDEX_PATH", base / "bm25.pkl"),
            patch.object(bm25_index, "METADATA_PATH", base / "bm25.meta.json"),
            patch.object(bm25_index, "DIRTY_MARKER_PATH", base / "bm25.dirty"),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._tmp.cleanup)
        for p in self._patches:
            self.addCleanup(p.stop)

    def test_rebuild_is_skipped_and_marked_dirty_inside_the_window(self):
        with patch.object(bm25_index, "_last_rebuild_at", return_value=1_000.0), patch.object(
            bm25_index, "time"
        ) as clock, patch.object(bm25_index, "_rebuild_now") as rebuild_now:
            clock.time.return_value = 1_010.0  # 10s later, inside a 60s window
            with patch.object(bm25_index, "BM25_REBUILD_MIN_INTERVAL_SECONDS", 60):
                result = bm25_index.rebuild_index_from_postgres()

        self.assertIsNone(result)
        rebuild_now.assert_not_called()
        self.assertTrue(bm25_index.is_dirty())

    def test_rebuild_runs_once_the_window_has_elapsed(self):
        with patch.object(bm25_index, "_last_rebuild_at", return_value=1_000.0), patch.object(
            bm25_index, "time"
        ) as clock, patch.object(bm25_index, "_rebuild_now") as rebuild_now:
            clock.time.return_value = 1_100.0  # 100s later, outside a 60s window
            with patch.object(bm25_index, "BM25_REBUILD_MIN_INTERVAL_SECONDS", 60):
                bm25_index.rebuild_index_from_postgres()

        rebuild_now.assert_called_once_with()

    def test_force_bypasses_the_debounce_window_entirely(self):
        # Operator-initiated work (backfill, admin reindex) must never be
        # silently skipped just because an ingestion just happened.
        with patch.object(bm25_index, "_last_rebuild_at", return_value=1_000.0), patch.object(
            bm25_index, "time"
        ) as clock, patch.object(bm25_index, "_rebuild_now") as rebuild_now:
            clock.time.return_value = 1_001.0
            with patch.object(bm25_index, "BM25_REBUILD_MIN_INTERVAL_SECONDS", 60):
                bm25_index.rebuild_index_from_postgres(force=True)

        rebuild_now.assert_called_once_with()

    def test_flush_rebuilds_only_when_dirty(self):
        with patch.object(bm25_index, "_rebuild_now") as rebuild_now:
            self.assertIsNone(bm25_index.flush_if_dirty())
            rebuild_now.assert_not_called()

            bm25_index._mark_dirty()
            bm25_index.flush_if_dirty()
            rebuild_now.assert_called_once_with()

    def test_recording_a_rebuild_clears_the_dirty_marker(self):
        bm25_index._mark_dirty()
        self.assertTrue(bm25_index.is_dirty())

        bm25_index._record_rebuild(1_234.5)

        self.assertFalse(bm25_index.is_dirty())
        self.assertEqual(bm25_index._last_rebuild_at(), 1_234.5)

    def test_missing_or_corrupt_metadata_reports_never_rebuilt(self):
        # A missing/corrupt metadata file must read as "never rebuilt" (0.0)
        # so the next call rebuilds rather than skipping forever.
        self.assertEqual(bm25_index._last_rebuild_at(), 0.0)

        bm25_index.METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        bm25_index.METADATA_PATH.write_text("{not valid json")
        self.assertEqual(bm25_index._last_rebuild_at(), 0.0)


if __name__ == "__main__":
    unittest.main()
