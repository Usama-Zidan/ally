"""
Google Drive integration. Mocked for now -- see jira_tool.py's module
docstring for the general pattern (real connected-accounts check, fake
response body, drop-in extension point for the real API).
"""
from __future__ import annotations

import uuid
from typing import Any

from services.agent.connected_accounts import get_connected_account
from services.agent.tools.base import Tool, ToolError, ToolResult
from services.agent.tools.contracts import validate_tool_arguments


class DriveTool(Tool):
    name = "drive"
    description = "Search, read, and create files in Google Drive."
    # Param names/types are spelled out explicitly here because the
    # classifier only ever sees this description string -- a vague
    # description leaves it to guess the actual key names (see
    # local_pc_tool.py's actions dict for the bug this pattern fixes).
    actions = {
        "search_files": "Search for files/folders. Params: query (string, required).",
        "get_file": "Get a file's metadata. Params: file_id (string, required).",
        "create_file": (
            "Create a new file (e.g. a Google Doc). Params: title (string, required), "
            "content (string, optional)."
        ),
    }

    def is_available(self, tenant_id: str) -> bool:
        return get_connected_account(tenant_id, "google_drive") is not None

    async def execute(self, tenant_id: str, action: str, params: dict[str, Any]) -> ToolResult:
        params = validate_tool_arguments(self.name, action, params)
        if get_connected_account(tenant_id, "google_drive") is None:
            raise ToolError(
                "Google Drive isn't connected yet. Connect a Google account before asking "
                "me to search or create files."
            )

        if action == "search_files":
            query = params.get("query", "")
            return ToolResult(
                success=True,
                summary=f"Found 0 files matching {query!r} (mocked -- no real Drive connection yet)",
                details={"mocked": True, "query": query, "results": []},
            )

        if action == "get_file":
            file_id = params.get("file_id")
            if not file_id:
                raise ToolError("get_file requires a 'file_id'.")
            return ToolResult(
                success=True,
                summary=f"File {file_id}: (mocked) Example Document",
                details={"mocked": True, "file_id": file_id, "name": "Example Document"},
            )

        if action == "create_file":
            title = params.get("title")
            if not title:
                raise ToolError("create_file requires a 'title'.")
            file_id = f"mock-{uuid.uuid4().hex[:10]}"
            return ToolResult(
                success=True,
                summary=f"Created Drive file {title!r}",
                details={
                    "mocked": True,
                    "file_id": file_id,
                    "title": title,
                    "content_preview": (params.get("content") or "")[:200],
                },
            )

        raise ToolError(f"Unknown Drive action {action!r}. Known actions: {list(self.actions)}")
