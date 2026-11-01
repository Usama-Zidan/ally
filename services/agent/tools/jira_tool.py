"""
Jira integration. Mocked for now -- every action returns a simulated but
realistically-shaped response rather than calling the real Jira API,
since no real Atlassian OAuth app/credentials exist yet.

What's already real: the connected-accounts check (is_available /
execute both go through services.agent.connected_accounts, so this tool
correctly reports "not connected" until an account is actually stored
there) and the action/param shape. Swapping in the real API later means
replacing the body of each `if action == ...` branch with an actual
`httpx` call to `{JIRA_BASE_URL}/rest/api/3/...` using
`account.access_token` -- the call sites in services.agent.graph and the
classifier prompt do not need to change.
"""
from __future__ import annotations

import uuid
from typing import Any

from config import JIRA_BASE_URL
from services.agent.connected_accounts import get_connected_account
from services.agent.tools.base import Tool, ToolError, ToolResult
from services.agent.tools.contracts import validate_tool_arguments


class JiraTool(Tool):
    name = "jira"
    description = "Create, search, and update Jira tickets."
    # Param names/types are spelled out explicitly here because the
    # classifier only ever sees this description string -- a vague
    # description leaves it to guess the actual key names (see
    # local_pc_tool.py's actions dict for the bug this pattern fixes).
    actions = {
        "create_ticket": (
            "Create a new Jira ticket. Params: summary (string, required), "
            "description (string, optional)."
        ),
        "search_tickets": "Search for tickets. Params: query (string, required).",
        "get_ticket": (
            "Get the details of a ticket. Params: ticket_key (string, required, e.g. 'PROJ-123')."
        ),
        "add_comment": (
            "Add a comment to an existing ticket. Params: ticket_key (string, required), "
            "comment (string, required)."
        ),
    }

    def is_available(self, tenant_id: str) -> bool:
        return get_connected_account(tenant_id, "jira") is not None

    async def execute(self, tenant_id: str, action: str, params: dict[str, Any]) -> ToolResult:
        params = validate_tool_arguments(self.name, action, params)
        account = get_connected_account(tenant_id, "jira")
        if account is None:
            raise ToolError(
                "Jira isn't connected yet. Connect a Jira account before asking me to "
                "create or look up tickets."
            )

        if action == "create_ticket":
            summary = params.get("summary")
            if not summary:
                raise ToolError("create_ticket requires a 'summary'.")
            ticket_key = f"MOCK-{uuid.uuid4().hex[:5].upper()}"
            base_url = JIRA_BASE_URL or "https://your-domain.atlassian.net"
            return ToolResult(
                success=True,
                summary=f"Created Jira ticket {ticket_key}: {summary}",
                details={
                    "mocked": True,
                    "ticket_key": ticket_key,
                    "summary": summary,
                    "description": params.get("description", ""),
                    "url": f"{base_url}/browse/{ticket_key}",
                },
            )

        if action == "search_tickets":
            query = params.get("query", "")
            return ToolResult(
                success=True,
                summary=f"Found 0 tickets matching {query!r} (mocked -- no real Jira connection yet)",
                details={"mocked": True, "query": query, "results": []},
            )

        if action == "get_ticket":
            ticket_key = params.get("ticket_key")
            if not ticket_key:
                raise ToolError("get_ticket requires a 'ticket_key'.")
            return ToolResult(
                success=True,
                summary=f"Ticket {ticket_key}: (mocked) Example ticket summary",
                details={
                    "mocked": True,
                    "ticket_key": ticket_key,
                    "status": "Open",
                    "summary": "Example ticket summary",
                },
            )

        if action == "add_comment":
            ticket_key = params.get("ticket_key")
            comment = params.get("comment")
            if not ticket_key or not comment:
                raise ToolError("add_comment requires 'ticket_key' and 'comment'.")
            return ToolResult(
                success=True,
                summary=f"Added a comment to {ticket_key}",
                details={"mocked": True, "ticket_key": ticket_key, "comment": comment},
            )

        raise ToolError(f"Unknown Jira action {action!r}. Known actions: {list(self.actions)}")
