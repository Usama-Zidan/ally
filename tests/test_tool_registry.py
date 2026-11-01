"""Tests for services.agent.tools.base: Tool, ToolResult, ToolRegistry."""
from __future__ import annotations

import unittest

from services.agent.tools.base import Tool, ToolRegistry, ToolResult


class _ToolA(Tool):
    name = "a"
    description = "Tool A does things"
    actions = {"act1": "does thing 1", "act2": "does thing 2"}

    async def execute(self, tenant_id, action, params):
        return ToolResult(success=True, summary="ok")


class _ToolB(Tool):
    """Only available for one specific tenant, to exercise is_available()
    gating -- this is the pattern every real tool (Jira/Drive/Calendar's
    connected-account check, LocalPC's live-connection check) follows."""

    name = "b"
    description = "Tool B does other things"
    actions = {"act3": "does thing 3"}

    def is_available(self, tenant_id: str) -> bool:
        return tenant_id == "connected-tenant"

    async def execute(self, tenant_id, action, params):
        return ToolResult(success=True, summary="ok")


class ToolRegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(_ToolA())
        self.registry.register(_ToolB())

    def test_get_returns_registered_tool_by_name(self):
        tool = self.registry.get("a")
        self.assertIsNotNone(tool)
        assert tool is not None
        self.assertEqual(tool.name, "a")

    def test_get_returns_none_for_unknown_name(self):
        self.assertIsNone(self.registry.get("nonexistent"))

    def test_all_returns_every_registered_tool_regardless_of_availability(self):
        names = {tool.name for tool in self.registry.all()}
        self.assertEqual(names, {"a", "b"})

    def test_available_for_excludes_tools_unavailable_to_this_tenant(self):
        available = self.registry.available_for("some-other-tenant")
        self.assertEqual([tool.name for tool in available], ["a"])

    def test_available_for_includes_tools_available_to_this_tenant(self):
        available = self.registry.available_for("connected-tenant")
        self.assertEqual({tool.name for tool in available}, {"a", "b"})

    def test_describe_for_classifier_lists_only_available_tools_and_their_actions(self):
        description = self.registry.describe_for_classifier("some-other-tenant")
        self.assertIn("a: Tool A does things", description)
        self.assertIn("act1", description)
        self.assertIn("act2", description)
        self.assertNotIn("b:", description)  # unavailable for this tenant

    def test_describe_for_classifier_handles_no_available_tools(self):
        empty_registry = ToolRegistry()
        description = empty_registry.describe_for_classifier("any-tenant")
        self.assertIn("no tools", description.lower())

    def test_registering_a_tool_with_the_same_name_replaces_the_previous_one(self):
        replacement = _ToolA()
        self.registry.register(replacement)
        self.assertIs(self.registry.get("a"), replacement)
        self.assertEqual(len(self.registry.all()), 2)  # still just a and b


class ToolResultTest(unittest.TestCase):
    def test_mock_results_are_visibly_simulated(self):
        result = ToolResult(success=True, summary="Created ticket", details={"mocked": True})
        self.assertTrue(result.summary.startswith("[SIMULATED]"))

    def test_details_defaults_to_an_empty_dict_not_shared_between_instances(self):
        # A mutable default (dataclass field(default_factory=dict)) must
        # not be accidentally shared across instances -- mutating one
        # ToolResult's details must never leak into another's.
        first = ToolResult(success=True, summary="a")
        second = ToolResult(success=True, summary="b")
        first.details["x"] = 1
        self.assertEqual(second.details, {})


if __name__ == "__main__":
    unittest.main()
