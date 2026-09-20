"""Tests for sentence-aware chunking with overlap.

Chunking quality sets a hard ceiling on retrieval quality that no amount
of reranking can recover, so these tests pin the properties that matter:
chunks never span pages, never exceed max_chars, never split a word, and
consecutive chunks share overlap so a fact stated at a boundary stays
retrievable.
"""
from __future__ import annotations

import asyncio
import itertools
import unittest

from workflows.activities import _pack_sentences, _split_into_sentences, chunk_pages

PARAGRAPH = " ".join(
    f"This is sentence number {index} and it carries a reasonable amount of policy text."
    for index in range(60)
)


class SentenceSplittingTest(unittest.TestCase):
    def test_splits_on_sentence_punctuation(self):
        sentences = _split_into_sentences("First one. Second one! Third one?")
        self.assertEqual(sentences, ["First one.", "Second one!", "Third one?"])

    def test_unpunctuated_text_still_yields_a_unit(self):
        # OCR output frequently lacks terminal punctuation; it must still
        # chunk rather than silently producing nothing.
        self.assertEqual(_split_into_sentences("no punctuation here"), ["no punctuation here"])

    def test_blank_text_yields_no_sentences(self):
        self.assertEqual(_split_into_sentences("   \n\n  "), [])


class PackSentencesTest(unittest.TestCase):
    def test_overlap_never_pushes_a_chunk_past_max_chars(self):
        # Regression: the overlap carried into a new chunk was previously
        # capped only by overlap_chars, ignoring how much room the
        # triggering sentence had already claimed -- so a generous
        # overlap_chars produced chunks longer than max_chars.
        for max_chars, overlap in itertools.product([80, 150, 400, 1500], [0, 20, 50, 200]):
            if overlap >= max_chars:
                continue
            with self.subTest(max_chars=max_chars, overlap=overlap):
                chunks = _pack_sentences(_split_into_sentences(PARAGRAPH), max_chars, overlap)
                multi_sentence = [c for c in chunks if len([s for s in c.split(". ") if s.strip()]) > 1]
                for chunk in multi_sentence:
                    self.assertLessEqual(len(chunk), max_chars)

    def test_consecutive_chunks_share_overlapping_text(self):
        chunks = _pack_sentences(_split_into_sentences(PARAGRAPH), 1500, 200)
        self.assertGreater(len(chunks), 1, "fixture should produce multiple chunks")

        for index in range(len(chunks) - 1):
            tail = [s for s in chunks[index].split(". ") if s.strip()][-1].strip().rstrip(".")
            self.assertIn(
                tail,
                chunks[index + 1],
                "a fact at a chunk boundary must remain retrievable from the next chunk",
            )

    def test_oversized_sentence_is_wrapped_on_word_boundaries(self):
        sentence = " ".join(["word"] * 200)  # single unpunctuated run
        chunks = _pack_sentences([sentence], max_chars=50, overlap_chars=10)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # Splitting inside a word is never useful for retrieval.
            for token in chunk.split(" "):
                self.assertEqual(token, "word")

    def test_no_overlap_requested_still_produces_valid_chunks(self):
        chunks = _pack_sentences(_split_into_sentences(PARAGRAPH), 200, 0)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)


class ChunkPagesTest(unittest.TestCase):
    def test_chunks_never_span_page_boundaries(self):
        pages = [
            {"page_number": 1, "text": "Leave is twenty-one days. Requests go via the portal."},
            {"page_number": 2, "text": "Probation lasts three months. It may be extended once."},
        ]
        chunks = asyncio.run(chunk_pages(pages, max_chars=60, overlap_chars=20))

        self.assertEqual({c["page_number"] for c in chunks}, {1, 2})
        for chunk in chunks:
            if chunk["page_number"] == 1:
                self.assertNotIn("Probation", chunk["text"])
            else:
                self.assertNotIn("Leave is", chunk["text"])

    def test_empty_pages_produce_no_chunks(self):
        chunks = asyncio.run(chunk_pages([{"page_number": 1, "text": "   "}]))
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
