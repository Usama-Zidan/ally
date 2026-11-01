"""
Google Calendar integration. Mocked for now -- see jira_tool.py's module
docstring for the general pattern.
"""
from __future__ import annotations

import uuid
from typing import Any

from services.agent.connected_accounts import get_connected_account
from services.agent.tools.base import Tool, ToolError, ToolResult


class CalendarTool(Tool):
    name = "calendar"
    description = "Create, list, and cancel Google Calendar events."
    # Param names/types are spelled out explicitly here because the
    # classifier only ever sees this description string -- a vague
    # description leaves it to guess the actual key names (see
    # local_pc_tool.py's actions dict for the bug this pattern fixes).
    actions = {
        "create_event": (
            "Create a new calendar event. Params: title (string, required), "
            "start_time (string, required, ISO 8601), end_time (string, optional, ISO 8601)."
        ),
        "list_events": (
            "List upcoming events in a date range. Params: start (string, optional, ISO 8601), "
            "end (string, optional, ISO 8601)."
        ),
        "delete_event": "Cancel an event. Params: event_id (string, required).",
    }

    def is_available(self, tenant_id: str) -> bool:
        return get_connected_account(tenant_id, "google_calendar") is not None

    async def execute(self, tenant_id: str, action: str, params: dict[str, Any]) -> ToolResult:
        if get_connected_account(tenant_id, "google_calendar") is None:
            raise ToolError(
                "Google Calendar isn't connected yet. Connect a Google account before "
                "asking me to manage your calendar."
            )

        if action == "create_event":
            title = params.get("title")
            start_time = params.get("start_time")
            if not title or not start_time:
                raise ToolError("create_event requires 'title' and 'start_time'.")
            event_id = f"mock-{uuid.uuid4().hex[:10]}"
            return ToolResult(
                success=True,
                summary=f"Created event {title!r} at {start_time}",
                details={
                    "mocked": True,
                    "event_id": event_id,
                    "title": title,
                    "start_time": start_time,
                    "end_time": params.get("end_time"),
                },
            )

        if action == "list_events":
            return ToolResult(
                success=True,
                summary="No upcoming events found (mocked -- no real Calendar connection yet)",
                details={
                    "mocked": True,
                    "start": params.get("start"),
                    "end": params.get("end"),
                    "results": [],
                },
            )

        if action == "delete_event":
            event_id = params.get("event_id")
            if not event_id:
                raise ToolError("delete_event requires an 'event_id'.")
            return ToolResult(
                success=True,
                summary=f"Cancelled event {event_id}",
                details={"mocked": True, "event_id": event_id},
            )

        raise ToolError(f"Unknown Calendar action {action!r}. Known actions: {list(self.actions)}")
