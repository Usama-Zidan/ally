"""Tests for services.chat.prompt."""
from __future__ import annotations

import unittest

from services.chat.prompt import SYSTEM_PROMPT, build_context_block, build_messages


SOURCES = [
    {"filename": "hr_policy.pdf", "page_number": 3, "text": "Employees receive 21 days of leave."},
    {"filename": "leave_policy.pdf", "page_number": 2, "text": "Requests go through the portal."},
]


class BuildContextBlockTest(unittest.TestCase):
    def test_numbers_sources_starting_at_one_with_filename_and_page(self):
        block = build_context_block(SOURCES)
        self.assertEqual(
            block,
            "[1] (hr_policy.pdf, page 3): Employees receive 21 days of leave.\n\n"
            "[2] (leave_policy.pdf, page 2): Requests go through the portal.",
        )

    def test_empty_sources_produces_empty_block(self):
        self.assertEqual(build_context_block([]), "")


class BuildMessagesTest(unittest.TestCase):
    def test_leads_with_system_prompt(self):
        messages = build_messages([], SOURCES, "How much leave do I get?")
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})

    def test_final_message_contains_numbered_sources_and_the_question(self):
        messages = build_messages([], SOURCES, "How much leave do I get?")
        content = messages[-1]["content"]
        self.assertTrue(content.startswith("Sources:"))
        self.assertIn("[1]", content)
        self.assertIn("[2]", content)
        self.assertIn("How much leave do I get?", content)

    def test_history_is_inserted_between_system_prompt_and_current_question(self):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        messages = build_messages(history, SOURCES, "follow up")

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1:3], history)
        self.assertEqual(messages[-1]["role"], "user")

    def test_empty_sources_omit_sources_prefix_entirely(self):
        # An empty "Sources:" block would look like relevant context that
        # happens to be blank, rather than no context at all -- omit the
        # prefix so the distinction is unambiguous to the model.
        messages = build_messages([], [], "anything")
        self.assertEqual(messages[-1]["content"], "Question: anything")

    def test_citation_numbering_is_stable_regardless_of_history_length(self):
        # Citation markers in the answer (e.g. "[2]") must always refer to
        # the same source regardless of how much prior history is replayed
        # -- numbering comes only from the sources list, never shifted by
        # history length.
        short_history = [{"role": "user", "content": "hi"}]
        long_history = short_history * 5

        content_short = build_messages(short_history, SOURCES, "q")[-1]["content"]
        content_long = build_messages(long_history, SOURCES, "q")[-1]["content"]
        self.assertEqual(content_short, content_long)


if __name__ == "__main__":
    unittest.main()
