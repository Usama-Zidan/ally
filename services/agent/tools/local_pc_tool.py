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

from typing import Any

from services.agent.pc_agent_bridge import PCAgentError, bridge
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
            "Open an application OR website by its everyday name. Params: app_name (string, "
            "required) -- e.g. 'spotify', 'edge', 'notepad', 'youtube', or a URL. "
            "The PC agent resolves the name itself (installed apps, then websites), "
            "so pass the name the user said, not a shell command."
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
            "timeout_seconds (number, optional)."
        ),
        "move_mouse": "Move the mouse cursor. Params: x (integer, required), y (integer, required).",
        "click": "Click the mouse at its current position. Params: button (string, optional: 'left'/'right'/'middle', default 'left').",
        "type_text": "Type text at the current cursor/focus position. Params: text (string, required).",
        "press_key": "Press a single key or key combination. Params: key (string, required, e.g. 'enter', 'ctrl+c').",
    }

    def is_available(self, tenant_id: str) -> bool:
        return bridge.is_connected(tenant_id)

    async def execute(self, tenant_id: str, action: str, params: dict[str, Any]) -> ToolResult:
        capability = _ACTION_CAPABILITY.get(action)
        if capability is None:
            raise ToolError(f"Unknown local PC action {action!r}. Known actions: {list(self.actions)}")

        try:
            details = await bridge.send_command(tenant_id, capability, action, params)
        except PCAgentError as exc:
            raise ToolError(str(exc)) from exc

        return ToolResult(success=True, summary=_summarize(action, params, details), details=details)


def _summarize(action: str, params: dict[str, Any], details: dict[str, Any]) -> str:
    """One-line human-readable summary for the chat transcript -- the
    full structured result is still available in ToolResult.details."""
    if action == "open_app":
        return f"Opened {params.get('app_name', 'the application')}"
    if action == "run_command":
        return f"Ran command: {params.get('command', '')}"
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
