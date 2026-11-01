"""Validate explicit PC commands while permitting classifier-routed requests."""

from __future__ import annotations

from typing import Any

from services.agent.commands import parse_command
from services.agent.tools.base import ToolError


def validate_explicit_pc_command(
    query: str, tool_name: str, action: str, params: dict[str, Any]
) -> None:
    """If the user supplied a slash command, bind dispatch to its exact content.

    Natural-language PC requests may be routed by the classifier. The companion
    remains the authorization boundary and requires consent for the exact action.
    """
    if tool_name != "local_pc":
        return
    command = parse_command(query)
    if command is None:
        return
    if command.tool_name != tool_name or command.action != action or command.params != params:
        raise ToolError("The selected PC action does not match the explicit command.")


# Keep the former internal name importable for callers while its behavior now
# permits natural-language routing.
require_explicit_pc_command = validate_explicit_pc_command
