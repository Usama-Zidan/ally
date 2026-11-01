"""
Tests for services.agent.graph.

These exercise the real, compiled LangGraph StateGraph (not a mock of
it) -- only the classifier and tool registry are patched, so the actual
routing/dispatch wiring (add_conditional_edges, the tool_node error
handling) is genuinely under test.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from services.agent import classifier as classifier_mod
from services.agent import registry as registry_mod
from services.agent.classifier import Intent
from services.agent.graph import build_graph, route_message as route_graph_message
from services.agent.tools.base import Tool, ToolError, ToolResult
from services.auth.principal import Principal, LEGACY_SCOPES


async def route_message(tenant_id, query):
    """Supply the authenticated credential required by tool-dispatch tests."""
    return await route_graph_message(
        tenant_id, query, principal=Principal(tenant_id, "test-credential", LEGACY_SCOPES)
    )


class _StubTool(Tool):
    name = "jira"
    description = "stub"
    actions = {"create_ticket": "stub"}

    def __init__(self, result=None, error=None, raise_unexpected=False):
        self._result = result
        self._error = error
        self._raise_unexpected = raise_unexpected

    async def execute(self, tenant_id, action, params):
        if self._raise_unexpected:
            raise RuntimeError("boom")
        if self._error:
            raise ToolError(self._error)
        return self._result


def _install_registry(*tools: Tool) -> None:
    registry_mod._registry = registry_mod.ToolRegistry()
    for tool in tools:
        registry_mod._registry.register(tool)


def _patch_classifier(intent: Intent):
    async def fake_classify(self, query, tools_block):
        return intent

    return patch.object(classifier_mod.LLMIntentClassifier, "classify", fake_classify)


class RouteMessageTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        registry_mod.reset_registry()

    async def test_explicit_command_bypasses_the_classifier_entirely(self):
        called = {"n": 0}

        async def fake_classify(self, query, tools_block):
            called["n"] += 1
            return Intent(kind="rag")

        _install_registry(_StubTool(result=ToolResult(success=True, summary="Created MOCK-1")))

        with patch.object(classifier_mod.LLMIntentClassifier, "classify", fake_classify):
            state = await route_message("tenant-a", '/jira.create_ticket summary="Fix bug"')

        self.assertEqual(state["route"], "tool")
        self.assertEqual(
            called["n"], 0, "classifier must not run when an explicit command is given"
        )
        self.assertEqual(state["intent"].params, {"summary": "Fix bug"})
        self.assertEqual(state["tool_result"].summary, "Created MOCK-1")

    async def test_compiled_graph_uses_its_injected_classifier_and_registry(self):
        tool = _StubTool(result=ToolResult(success=True, summary="isolated result"))
        registry = registry_mod.ToolRegistry()
        registry.register(tool)

        class Classifier:
            async def classify(self, query, tools_block):
                self.asserted_tools = tools_block
                return Intent(kind="tool", tool_name="jira", action="create_ticket")

        classifier = Classifier()
        graph = build_graph(classifier=classifier, registry=registry)
        state = await route_graph_message(
            "tenant-a",
            "please do this",
            principal=Principal("tenant-a", "injected-credential", LEGACY_SCOPES),
            graph=graph,
        )

        self.assertEqual(state["tool_result"].summary, "isolated result")
        self.assertIn("jira", classifier.asserted_tools)

    async def test_classifier_routes_to_rag_without_executing_any_tool(self):
        _install_registry()
        with _patch_classifier(Intent(kind="rag")):
            state = await route_message("tenant-a", "What is our leave policy?")

        self.assertEqual(state["route"], "rag")
        self.assertNotIn("tool_result", state)
        self.assertNotIn("tool_error", state)

    async def test_classifier_routes_to_conversational(self):
        _install_registry()
        with _patch_classifier(Intent(kind="conversational")):
            state = await route_message("tenant-a", "hey, how's it going")

        self.assertEqual(state["route"], "conversational")

    async def test_classifier_routing_to_an_unregistered_tool_is_a_clean_error(self):
        _install_registry()  # empty -- "jira" is not registered
        with _patch_classifier(
            Intent(kind="tool", tool_name="jira", action="create_ticket", params={})
        ):
            state = await route_message("tenant-a", "create a jira ticket")

        self.assertEqual(state["route"], "tool")
        self.assertIn("Unknown tool", state["tool_error"])
        self.assertNotIn("tool_result", state)

    async def test_tool_error_surfaces_cleanly_not_as_a_crash(self):
        _install_registry(_StubTool(error="Jira isn't connected yet."))
        with _patch_classifier(
            Intent(kind="tool", tool_name="jira", action="create_ticket", params={})
        ):
            state = await route_message("tenant-a", "create a jira ticket")

        self.assertEqual(state["tool_error"], "Jira isn't connected yet.")

    async def test_unexpected_tool_exception_is_caught_not_propagated(self):
        # A bug inside a tool implementation (not an expected ToolError)
        # must still come back as a clean tool_error, never an unhandled
        # exception that would crash the chat turn.
        _install_registry(_StubTool(raise_unexpected=True))
        with _patch_classifier(
            Intent(kind="tool", tool_name="jira", action="create_ticket", params={})
        ):
            state = await route_message("tenant-a", "create a jira ticket")

        self.assertIn("failed unexpectedly", state["tool_error"])

    async def test_missing_action_on_a_tool_intent_is_a_clean_error(self):
        _install_registry(_StubTool())
        with _patch_classifier(Intent(kind="tool", tool_name="jira", action=None, params={})):
            state = await route_message("tenant-a", "do something with jira")

        self.assertIn("No action specified", state["tool_error"])


if __name__ == "__main__":
    unittest.main()
