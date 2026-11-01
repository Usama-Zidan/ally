"""
The Tool interface every agent integration implements, and the registry
that makes "might add new apps or integrations in the future" a matter of
writing one new class and registering it -- not restructuring the graph,
the classifier prompt, or the WebSocket protocol.

Adding a new tool:
    1. Subclass Tool, implement `execute()`.
    2. Add an instance to `build_default_registry()` in registry.py.
That's it -- the classifier learns about it from `describe()`, and
services.agent.graph dispatches to it by name automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from services.auth.principal import AuthorizationDenied, Principal, authorize_tool


@dataclass
class ToolResult:
    """The outcome of one tool action, in a shape the chat WebSocket
    protocol can render directly (see services.api.main's {"type":
    "tool_result", ...} message)."""

    success: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.details.get("mocked") is True:
            self.summary = f"[SIMULATED] {self.summary}"


class ToolError(Exception):
    """Raised by a Tool when an action can't be completed (not connected,
    invalid params, the remote API rejected the request, etc.). Distinct
    from an unhandled exception so services.agent.graph can turn it into
    a clean, user-facing error message instead of a stack trace."""


class Tool(ABC):
    """One integration (Jira, Drive, Calendar, the local PC agent, ...).

    `name` is the identifier used everywhere else: the classifier's
    output, the explicit slash-command prefix (services.agent.commands),
    and dispatch in services.agent.graph. Keep it stable once used --
    renaming it is a breaking change for any saved command shortcuts.
    """

    name: str
    description: str  # Shown to the classifier so it knows when to pick this tool.
    actions: dict[str, str]  # action_name -> one-line description, for the classifier/help text.

    @abstractmethod
    async def execute(self, tenant_id: str, action: str, params: dict[str, Any]) -> ToolResult:
        """Performs ``action`` with ``params`` on behalf of ``tenant_id``.

        Raises:
            ToolError: For any expected failure (not connected, invalid
                action/params, the remote call failed) -- never lets a
                raw exception from an HTTP client or similar propagate.
        """

    def is_available(self, tenant_id: str) -> bool:
        """Whether this tool can actually be used right now for this
        tenant (e.g. an account is connected, or a PC agent is online).
        Tools that need no per-tenant setup can leave this as True.
        """
        return True


class ToolRegistry:
    """Holds every registered Tool, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def available_for(self, tenant_id: str) -> list[Tool]:
        """Tools usable right now for this tenant -- what the classifier
        should actually be allowed to route to, as opposed to every tool
        that exists in the codebase."""
        return [tool for tool in self._tools.values() if tool.is_available(tenant_id)]

    def describe_for_classifier(self, tenant_id: str) -> str:
        """Renders the available tools as a block the classifier prompt
        can include, so routing decisions are always grounded in what's
        actually usable for this tenant right now."""
        tools = self.available_for(tenant_id)
        if not tools:
            return "(no tools are currently connected for this user)"

        lines = []
        for tool in tools:
            actions = ", ".join(tool.actions)
            lines.append(f"- {tool.name}: {tool.description} (actions: {actions})")
        return "\n".join(lines)

    def describe_for_principal(self, principal: Principal) -> str:
        """Describe only permitted actions, without probing unauthorized accounts."""
        lines = []
        for tool in self._tools.values():
            actions = []
            for action in tool.actions:
                try:
                    authorize_tool(principal, principal.tenant_id, tool.name, action)
                except AuthorizationDenied:
                    continue
                actions.append(action)
            if actions and tool.is_available(principal.tenant_id):
                lines.append(f"- {tool.name}: {tool.description} (actions: {', '.join(actions)})")
        return "\n".join(lines) or "(no tools are authorized and connected for this credential)"
