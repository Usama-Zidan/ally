"""Tests for services.agent.classifier."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from services.agent import classifier as classifier_mod
from services.agent.classifier import (
    Intent,
    LLMIntentClassifier,
    _parse_intent,
    get_classifier,
    reset_classifier,
)


class ParseIntentTest(unittest.TestCase):
    def test_parses_a_clean_tool_intent(self):
        intent = _parse_intent(
            '{"kind": "tool", "tool_name": "jira", "action": "create_ticket", '
            '"params": {"summary": "x"}}'
        )
        self.assertEqual(intent.kind, "tool")
        self.assertEqual(intent.tool_name, "jira")
        self.assertEqual(intent.action, "create_ticket")
        self.assertEqual(intent.params, {"summary": "x"})

    def test_parses_a_rag_intent_with_null_tool_fields(self):
        intent = _parse_intent('{"kind": "rag", "tool_name": null, "action": null, "params": {}}')
        self.assertEqual(intent.kind, "rag")
        self.assertIsNone(intent.tool_name)

    def test_strips_markdown_json_fences(self):
        intent = _parse_intent('```json\n{"kind": "conversational", "params": {}}\n```')
        self.assertEqual(intent.kind, "conversational")

    def test_strips_bare_triple_backtick_fences_without_json_marker(self):
        intent = _parse_intent('```\n{"kind": "rag", "params": {}}\n```')
        self.assertEqual(intent.kind, "rag")

    def test_unparseable_output_falls_back_to_rag_not_a_crash(self):
        intent = _parse_intent("I think the user wants to know about leave policy.")
        self.assertEqual(intent.kind, "rag")
        self.assertIn("fallback", intent.reasoning)

    def test_invalid_kind_value_falls_back_to_rag(self):
        intent = _parse_intent('{"kind": "do_something_weird"}')
        self.assertEqual(intent.kind, "rag")

    def test_missing_params_defaults_to_empty_dict(self):
        intent = _parse_intent('{"kind": "tool", "tool_name": "jira", "action": "get_ticket"}')
        self.assertEqual(intent.params, {})

    def test_empty_string_falls_back_to_rag(self):
        intent = _parse_intent("")
        self.assertEqual(intent.kind, "rag")


class LLMIntentClassifierTest(unittest.IsolatedAsyncioTestCase):
    async def test_classify_sends_tools_block_in_the_system_prompt(self):
        classifier = LLMIntentClassifier()
        with patch.object(
            classifier_mod,
            "complete_chat",
            new=AsyncMock(return_value='{"kind": "rag", "params": {}}'),
        ) as complete_chat:
            intent = await classifier.classify("What is our leave policy?", "- jira: ...")

        self.assertEqual(intent.kind, "rag")
        messages = complete_chat.call_args.args[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("- jira: ...", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "What is our leave policy?\n\n/no_think"})


class ClassifierRegistryTest(unittest.TestCase):
    def tearDown(self):
        reset_classifier()

    def test_default_classifier_is_llm_based(self):
        reset_classifier()
        with patch.object(classifier_mod, "INTENT_CLASSIFIER", "llm"):
            instance = get_classifier()
        self.assertIsInstance(instance, LLMIntentClassifier)

    def test_same_instance_is_reused_across_calls(self):
        reset_classifier()
        with patch.object(classifier_mod, "INTENT_CLASSIFIER", "llm"):
            first = get_classifier()
            second = get_classifier()
        self.assertIs(first, second)

    def test_unknown_classifier_name_raises_a_clear_error(self):
        reset_classifier()
        with patch.object(classifier_mod, "INTENT_CLASSIFIER", "jev"):
            with self.assertRaises(RuntimeError) as ctx:
                get_classifier()
        self.assertIn("jev", str(ctx.exception))

    def test_registering_a_new_classifier_makes_it_selectable(self):
        # Mirrors how a custom "JEV" classifier would be added later: one
        # new entry in _CLASSIFIERS, no changes to get_classifier() itself.
        class FakeClassifier(classifier_mod.IntentClassifier):
            async def classify(self, query, tools_block):
                return Intent(kind="conversational")

        classifier_mod._CLASSIFIERS["fake"] = FakeClassifier
        self.addCleanup(classifier_mod._CLASSIFIERS.pop, "fake")

        reset_classifier()
        with patch.object(classifier_mod, "INTENT_CLASSIFIER", "fake"):
            instance = get_classifier()
        self.assertIsInstance(instance, FakeClassifier)


if __name__ == "__main__":
    unittest.main()
