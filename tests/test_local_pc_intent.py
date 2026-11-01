"""Tests for natural-language local PC intent fallbacks."""

from __future__ import annotations

import unittest

from services.agent.local_pc_intent import infer_local_pc_intent, web_search_phrase


class InferLocalPCIntentTest(unittest.TestCase):
    def test_recognizes_local_pc_access_scope_question_as_conversational(self):
        intent = infer_local_pc_intent("Can you read files on my PC?")

        assert intent is not None
        self.assertEqual(intent.kind, "conversational")

    def test_recognizes_mouse_corner_request(self):
        intent = infer_local_pc_intent("Move my cursor to the lower left corner")

        assert intent is not None
        self.assertEqual(intent.kind, "tool")
        self.assertEqual(intent.tool_name, "local_pc")
        self.assertEqual(intent.action, "move_mouse_to_corner")
        self.assertEqual(intent.params, {"corner": "bottom_left"})

    def test_recognizes_specific_web_request_and_preserves_query(self):
        query = "Find a YouTube video about gardening"
        intent = infer_local_pc_intent(query)

        assert intent is not None
        self.assertEqual(intent.kind, "tool")
        self.assertEqual(intent.action, "open_app")
        self.assertEqual(
            intent.params,
            {"app_name": "youtube", "search_query": query},
        )

    def test_does_not_infer_tool_for_unrelated_search(self):
        self.assertIsNone(infer_local_pc_intent("Find the customer policy"))


class WebSearchPhraseTest(unittest.TestCase):
    def test_does_not_use_site_only_request_as_search_phrase(self):
        self.assertIsNone(web_search_phrase("open youtube", "youtube"))

    def test_preserves_specific_media_request(self):
        query = "open the youtube video about gardening"

        self.assertEqual(web_search_phrase(query, "youtube"), query)


if __name__ == "__main__":
    unittest.main()
