"""Malformed tool arguments cannot trigger credential I/O or claim effects."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from services.agent.tools.base import ToolError
from services.agent.tools.calendar_tool import CalendarTool
from services.agent.tools.drive_tool import DriveTool
from services.agent.tools.jira_tool import JiraTool
from services.agent.tools.contracts import validate_tool_arguments


class CloudContractsTest(unittest.IsolatedAsyncioTestCase):
    async def test_each_adapter_rejects_types_unknown_fields_and_actions_before_account_lookup(
        self,
    ):
        for tool, module, action, valid in (
            (JiraTool(), "jira_tool", "create_ticket", {"summary": "x"}),
            (DriveTool(), "drive_tool", "create_file", {"title": "x"}),
            (
                CalendarTool(),
                "calendar_tool",
                "create_event",
                {"title": "x", "start_time": "2026-01-01T10:00:00Z"},
            ),
        ):
            for selected, payload in (
                (action, {**valid, "tenant_id": "victim"}),
                (action, {**valid, "confirmed": True}),
                (action, {key: [value] for key, value in valid.items()}),
                ("unrecognized", valid),
            ):
                with (
                    self.subTest(tool=tool.name, action=selected, payload=payload),
                    patch(f"services.agent.tools.{module}.get_connected_account") as account,
                ):
                    with self.assertRaises(ToolError):
                        await tool.execute("tenant-a", selected, payload)
                    account.assert_not_called()

    async def test_search_requires_a_nonempty_bounded_string(self):
        for tool, action in (("jira", "search_tickets"), ("drive", "search_files")):
            for query in (None, "", "   ", "x" * 2049, 42, {}):
                with self.subTest(tool=tool, query=query), self.assertRaises(ToolError):
                    validate_tool_arguments(tool, action, {"query": query})

    async def test_identifiers_cannot_embed_urls_paths_or_query_fragments(self):
        for tool, action, key in (
            ("jira", "get_ticket", "ticket_key"),
            ("drive", "get_file", "file_id"),
            ("calendar", "delete_event", "event_id"),
        ):
            for value in (
                "../other",
                "https://attacker.invalid",
                "id?token=x",
                "id#fragment",
                "ABC-1\n",
                "ABC-1\r",
                "ABC-1\x00",
                "x" * 129,
            ):
                with self.subTest(tool=tool, value=value), self.assertRaises(ToolError):
                    validate_tool_arguments(tool, action, {key: value})

    async def test_calendar_timestamps_require_timezone_and_positive_chronological_range(self):
        for payload in (
            {"title": "meeting", "start_time": "2026-01-01T10:00:00"},
            {"title": "meeting", "start_time": "2026-01-01"},
            {"title": "meeting", "start_time": "not-a-timestamp"},
            {
                "title": "meeting",
                "start_time": "2026-01-01T10:00:00Z",
                "end_time": "2026-01-01T09:00:00Z",
            },
            {
                "title": "meeting",
                "start_time": "2026-01-01T10:00:00Z",
                "end_time": "2026-01-01T12:00:00+02:00",
            },
        ):
            with self.subTest(payload=payload), self.assertRaises(ToolError):
                validate_tool_arguments("calendar", "create_event", payload)
        with self.assertRaises(ToolError):
            validate_tool_arguments(
                "calendar",
                "list_events",
                {"start": "2026-01-02T10:00:00Z", "end": "2026-01-01T10:00:00Z"},
            )
        payload = {
            "title": "meeting",
            "start_time": "2026-01-01T10:00:00Z",
            "end_time": "2026-01-01T13:00:00+02:00",
        }
        self.assertEqual(validate_tool_arguments("calendar", "create_event", payload), payload)

    async def test_rejected_inputs_are_not_reflected_in_public_error(self):
        secret = "sensitive-sentinel-argument"
        with self.assertRaises(ToolError) as context:
            validate_tool_arguments("drive", "create_file", {"title": "x", "unexpected": secret})
        self.assertNotIn(secret, str(context.exception))


if __name__ == "__main__":
    unittest.main()
