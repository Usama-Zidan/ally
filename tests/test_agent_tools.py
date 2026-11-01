"""Tests for the mocked cloud-API tools: JiraTool, DriveTool, CalendarTool.

Each tool's credential gate goes through services.agent.connected_accounts,
so these tests patch that module's get_connected_account rather than the
database underneath it -- the vault itself is covered separately in
test_connected_accounts.py.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from services.agent.connected_accounts import ConnectedAccount
from services.agent.tools.base import ToolError
from services.agent.tools.calendar_tool import CalendarTool
from services.agent.tools.drive_tool import DriveTool
from services.agent.tools.jira_tool import JiraTool

_CONNECTED = ConnectedAccount(
    tenant_id="tenant-a",
    provider="jira",
    access_token="token",
    refresh_token=None,
    expires_at=None,
    scopes=[],
)


def _async(coro):
    import asyncio

    return asyncio.run(coro)


class JiraToolTest(unittest.TestCase):
    def setUp(self):
        self.tool = JiraTool()

    def test_is_available_reflects_whether_an_account_is_connected(self):
        with patch("services.agent.tools.jira_tool.get_connected_account", return_value=None):
            self.assertFalse(self.tool.is_available("tenant-a"))
        with patch(
            "services.agent.tools.jira_tool.get_connected_account", return_value=_CONNECTED
        ):
            self.assertTrue(self.tool.is_available("tenant-a"))

    def test_execute_raises_tool_error_when_not_connected(self):
        with patch("services.agent.tools.jira_tool.get_connected_account", return_value=None):
            with self.assertRaises(ToolError):
                _async(self.tool.execute("tenant-a", "create_ticket", {"summary": "x"}))

    def test_create_ticket_requires_a_summary(self):
        with patch(
            "services.agent.tools.jira_tool.get_connected_account", return_value=_CONNECTED
        ):
            with self.assertRaises(ToolError):
                _async(self.tool.execute("tenant-a", "create_ticket", {}))

    def test_create_ticket_returns_a_mocked_ticket_key(self):
        with patch(
            "services.agent.tools.jira_tool.get_connected_account", return_value=_CONNECTED
        ):
            result = _async(
                self.tool.execute(
                    "tenant-a", "create_ticket", {"summary": "Fix login bug"}
                )
            )
        self.assertTrue(result.success)
        self.assertTrue(result.details["ticket_key"].startswith("MOCK-"))
        self.assertTrue(result.details["mocked"])
        self.assertIn("Fix login bug", result.summary)

    def test_get_ticket_requires_a_ticket_key(self):
        with patch(
            "services.agent.tools.jira_tool.get_connected_account", return_value=_CONNECTED
        ):
            with self.assertRaises(ToolError):
                _async(self.tool.execute("tenant-a", "get_ticket", {}))

    def test_unknown_action_raises_tool_error(self):
        with patch(
            "services.agent.tools.jira_tool.get_connected_account", return_value=_CONNECTED
        ):
            with self.assertRaises(ToolError):
                _async(self.tool.execute("tenant-a", "delete_everything", {}))


class DriveToolTest(unittest.TestCase):
    def setUp(self):
        self.tool = DriveTool()

    def test_execute_raises_tool_error_when_not_connected(self):
        with patch("services.agent.tools.drive_tool.get_connected_account", return_value=None):
            with self.assertRaises(ToolError):
                _async(self.tool.execute("tenant-a", "search_files", {"query": "x"}))

    def test_create_file_requires_a_title(self):
        with patch(
            "services.agent.tools.drive_tool.get_connected_account", return_value=_CONNECTED
        ):
            with self.assertRaises(ToolError):
                _async(self.tool.execute("tenant-a", "create_file", {}))

    def test_create_file_returns_a_mocked_file_id(self):
        with patch(
            "services.agent.tools.drive_tool.get_connected_account", return_value=_CONNECTED
        ):
            result = _async(
                self.tool.execute(
                    "tenant-a", "create_file", {"title": "Meeting notes", "content": "..."}
                )
            )
        self.assertTrue(result.success)
        self.assertTrue(result.details["file_id"].startswith("mock-"))
        self.assertIn("Meeting notes", result.summary)


class CalendarToolTest(unittest.TestCase):
    def setUp(self):
        self.tool = CalendarTool()

    def test_execute_raises_tool_error_when_not_connected(self):
        with patch(
            "services.agent.tools.calendar_tool.get_connected_account", return_value=None
        ):
            with self.assertRaises(ToolError):
                _async(self.tool.execute("tenant-a", "list_events", {}))

    def test_create_event_requires_title_and_start_time(self):
        with patch(
            "services.agent.tools.calendar_tool.get_connected_account",
            return_value=_CONNECTED,
        ):
            with self.assertRaises(ToolError):
                _async(self.tool.execute("tenant-a", "create_event", {"title": "Standup"}))

    def test_create_event_returns_a_mocked_event_id(self):
        with patch(
            "services.agent.tools.calendar_tool.get_connected_account",
            return_value=_CONNECTED,
        ):
            result = _async(
                self.tool.execute(
                    "tenant-a",
                    "create_event",
                    {"title": "Standup", "start_time": "2026-01-01T09:00:00Z"},
                )
            )
        self.assertTrue(result.success)
        self.assertTrue(result.details["event_id"].startswith("mock-"))

    def test_delete_event_requires_an_event_id(self):
        with patch(
            "services.agent.tools.calendar_tool.get_connected_account",
            return_value=_CONNECTED,
        ):
            with self.assertRaises(ToolError):
                _async(self.tool.execute("tenant-a", "delete_event", {}))


if __name__ == "__main__":
    unittest.main()
