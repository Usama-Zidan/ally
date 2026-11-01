"""
The local Windows companion agent, reached through
services.agent.pc_agent_bridge rather than a direct API call -- see that
module's docstring for why the connection is inverted.

Every action maps to exactly one capability (open_apps, file_rw,
shell_exec, mouse_keyboard_control). The *actual* enforcement of those
capabilities happens on the local machine itself, inside local_agent/ --
this tool and the bridge only know whether the connected agent *reports*
a capability as enabled; they cannot force a local machine to do
anything its own operator hasn't independently allowed. See
local_agent/README.md for the enforcement side.

Deliberately not built in v1 (the user explicitly scoped this out):
screenshots, clipboard access, system info. Adding one later is a new
action under a new or existing capability, same pattern as everything
else here.
"""

from __future__ import annotations

import math
from typing import Any

from services.agent.pc_agent_bridge import PCAgentBridge, PCAgentError, bridge
from services.agent.tools.base import Tool, ToolError, ToolResult

# action_name -> which capability toggle gates it.
_ACTION_CAPABILITY = {
    "open_app": "open_apps",
    "list_windows": "open_apps",
    "focus_window": "open_apps",
    "close_window": "open_apps",
    "read_file": "file_rw",
    "write_file": "file_rw",
    "list_directory": "file_rw",
    "run_command": "shell_exec",
    "move_mouse": "mouse_keyboard_control",
    "move_mouse_to_corner": "mouse_keyboard_control",
    "click": "mouse_keyboard_control",
    "type_text": "mouse_keyboard_control",
    "press_key": "mouse_keyboard_control",
}


class LocalPCTool(Tool):
    name = "local_pc"
    description = (
        "Control the user's own Windows PC directly: open applications, read/write files "
        "in folders the user has allowed, run shell commands, or control the mouse and "
        "keyboard. Only available when the user's PC agent is connected and has the "
        "relevant capability enabled."
    )
    # Every action description spells out its exact param names, types,
    # and whether each is required -- the classifier only ever sees this
    # string (ToolRegistry.describe_for_classifier), so a vague
    # description like "open an application by name" leaves it to guess
    # the actual key (observed in practice: it invented "app" and "url"
    # instead of the real "app_name"). State the literal param names here
    # and the classifier uses them correctly.
    actions = {
        "open_app": (
            "Open an application or website by the name the user said. Params: app_name "
            "(string, required) -- e.g. 'spotify', 'edge', 'notepad', 'youtube', a channel/video "
            "title, or a URL. The PC agent resolves installed apps and known sites locally; "
            "if it cannot resolve the name, it searches the web from that PC, shows the top "
            "candidate title and URL in the local approval prompt, and opens it only after "
            "the user types APPROVE. For a specific video/channel on a known site, include "
            "search_query with the full user phrase so the PC searches instead of opening "
            "the site's home page. Pass the user's search phrase, not a shell command."
        ),
        "list_windows": "List currently open windows. No params.",
        "focus_window": "Bring a window to the foreground. Params: title (string, required).",
        "close_window": "Close a window. Params: title (string, required).",
        "read_file": (
            "Read a file's contents (must be in an allowed folder). "
            "Params: path (string, required, absolute path)."
        ),
        "write_file": (
            "Write/overwrite a file's contents (must be in an allowed folder). "
            "Params: path (string, required, absolute path), "
            "content (string, required, the full new file content)."
        ),
        "list_directory": (
            "List a directory's contents (must be an allowed folder). "
            "Params: path (string, required, absolute path)."
        ),
        "run_command": (
            "Run a shell command and return its output. "
            "Params: command (string, required, a Windows cmd.exe command -- "
            "this agent runs on Windows, not Linux/macOS, so use Windows syntax "
            "e.g. 'dir', 'echo text > file.txt', not 'ls' or Unix paths), "
            "cwd (string, optional, absolute working directory), "
            "timeout_seconds (number, optional). Use open_app for websites or GUI apps; "
            "this bounded shell action is for commands, not launching persistent apps."
        ),
        "move_mouse": "Move the mouse cursor. Params: x (integer, required), y (integer, required).",
        "move_mouse_to_corner": (
            "Move the mouse cursor to a screen corner. Params: corner (string, required; "
            "one of top_left, top_right, bottom_left, bottom_right). For 'rightmost corner' "
            "without a vertical direction, use top_right."
        ),
        "click": "Click the mouse at its current position. Params: button (string, optional: 'left'/'right'/'middle', default 'left').",
        "type_text": "Type text at the current cursor/focus position. Params: text (string, required).",
        "press_key": "Press a single key or key combination. Params: key (string, required, e.g. 'enter', 'ctrl+c').",
    }

    def __init__(self, pc_bridge: PCAgentBridge | None = None) -> None:
        self.bridge = pc_bridge or bridge

    def is_available(self, tenant_id: str) -> bool:
        return self.bridge.is_connected(tenant_id)

    async def execute(self, tenant_id: str, action: str, params: dict[str, Any]) -> ToolResult:
        capability = _ACTION_CAPABILITY.get(action)
        if capability is None:
            raise ToolError(
                f"Unknown local PC action {action!r}. Known actions: {list(self.actions)}"
            )

        # Slash commands carry strings. Normalize numeric fields before sending
        # them to the companion's independently validated typed boundary.
        params = dict(params)
        try:
            for key in ("x", "y") if action == "move_mouse" else ():
                value = params.get(key)
                if type(value) is not int:
                    if not isinstance(value, str):
                        raise ValueError("Invalid coordinate")
                    params[key] = int(value)
            if action == "run_command" and "timeout_seconds" in params:
                value = params["timeout_seconds"]
                if type(value) not in (str, int, float):
                    raise ValueError("Invalid timeout")
                timeout = float(value)
                if not math.isfinite(timeout) or not 0 < timeout <= 120:
                    raise ValueError("Invalid timeout")
                params["timeout_seconds"] = timeout
        except (ValueError, TypeError) as exc:
            raise ToolError("Invalid numeric PC action arguments.") from exc

        try:
            details = await self.bridge.send_command(tenant_id, capability, action, params)
        except PCAgentError as exc:
            raise ToolError(str(exc)) from exc

        summary = _summarize(action, params, details)
        if details.get("truncated") is True:
            summary += " [partial result: output limit reached]"
        success = True
        if action == "run_command":
            success = (
                details.get("outcome") == "exited"
                and type(details.get("returncode")) is int
                and details["returncode"] == 0
                and details.get("cleanup_confirmed") is True
                and details.get("audit_recorded") is True
            )
        return ToolResult(success=success, summary=summary, details=details)


def _summarize(action: str, params: dict[str, Any], details: dict[str, Any]) -> str:
    """One-line human-readable summary for the chat transcript -- the
    full structured result is still available in ToolResult.details."""
    if action == "open_app":
        if details.get("method") == "web_search":
            title = details.get("title", "website")
            url = details.get("launched", "URL unavailable")
            return f"Opened search result: {title} ({url})"
        return f"Opened {params.get('app_name', 'the application')}"
    if action == "run_command":
        outcome = details.get("outcome")
        if not isinstance(outcome, str):
            outcome = ""
        if outcome == "exited" and type(details.get("returncode")) is int:
            summary = (
                f"Shell exited with code {details['returncode']}; requested effects are unverified."
            )
        else:
            label = {
                "timed_out": "timed out",
                "cancelled": "was cancelled",
                "output_limit": "reached its output limit",
                "startup_failed": "could not start",
                "runtime_error": "encountered an execution error",
            }.get(outcome, "status is unknown")
            summary = f"Shell {label}; effects may be partial. Inspect the device before retrying."
        if details.get("cleanup_confirmed") is not True:
            summary += " Child-process cleanup is unconfirmed."
        if details.get("audit_recorded") is not True:
            summary += " Terminal audit is missing."
        return summary
    if action == "read_file":
        return f"Read {params.get('path', 'the file')}"
    if action == "write_file":
        return f"Wrote to {params.get('path', 'the file')}"
    if action == "list_directory":
        count = len(details.get("entries", []))
        return f"Listed {params.get('path', 'the directory')} ({count} entries)"
    if action == "list_windows":
        count = len(details.get("windows", []))
        return f"Found {count} open window(s)"
    return f"Completed {action}"
