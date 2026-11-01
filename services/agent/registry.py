"""Builds the default ToolRegistry. See services.agent.tools.base for
how to add a new tool -- this is the only file a new one needs to be
registered in."""
from __future__ import annotations

from services.agent.tools.base import ToolRegistry
from services.agent.tools.calendar_tool import CalendarTool
from services.agent.tools.drive_tool import DriveTool
from services.agent.tools.jira_tool import JiraTool
from services.agent.tools.local_pc_tool import LocalPCTool


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(JiraTool())
    registry.register(DriveTool())
    registry.register(CalendarTool())
    registry.register(LocalPCTool())
    return registry


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = build_default_registry()
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = None
