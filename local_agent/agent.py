"""
Local companion agent: runs on your own Windows PC, connects outbound to
the Ally backend over WebSocket, and executes commands the chat agent
sends it -- opening apps, reading/writing files, running shell commands,
or controlling the mouse/keyboard, depending on what YOU enable below.

THIS FILE IS THE ACTUAL SECURITY BOUNDARY, not the server. The backend's
`pc_agent_capabilities` table and the handshake capabilities this script
reports are a declaration, not an enforcement mechanism -- the server
cannot force your machine to do anything. Every capability check in this
script re-validates locally regardless of what the server believes is
enabled, because this script is the only thing that can actually stop an
action from happening.

Setup:
    1. pip install -r requirements.txt
    2. cp config.example.json config.json, then edit it:
       - server_url: your backend's wss:// address
       - api_key: an API key provisioned via `services.auth.provision_tenant`
       - capabilities: which of the four categories below to allow
       - allowed_folders: absolute paths file_rw is restricted to
    3. python agent.py

Capabilities (each independently toggleable in config.json):
    open_apps               -- launch applications, list/focus/close windows
    file_rw                 -- read/write files, ONLY inside allowed_folders
    shell_exec               -- run arbitrary shell commands (HIGH RISK: see README)
    mouse_keyboard_control   -- move the mouse, click, type, press keys

Optional dependencies degrade gracefully: if pygetwindow/pyautogui aren't
installed, their capabilities report a clear error instead of crashing
the whole agent, so you can use (for example) just file_rw without
installing packages the other capabilities need.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import websockets

import app_launcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ally-local-agent")

# A dedicated, append-only audit log for the highest-risk capability.
# Always written when shell_exec is enabled, regardless of the general
# logging level above -- a record of every command this agent has ever
# run on your behalf should not be optional.
_AUDIT_LOG_PATH = Path(__file__).parent / "shell_exec_audit.log"


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("capabilities", {})
    for capability in ("open_apps", "file_rw", "shell_exec", "mouse_keyboard_control"):
        config["capabilities"].setdefault(capability, False)
    config.setdefault("allowed_folders", [])
    config.setdefault("app_aliases", {})
    return config


def _resolve_allowed_path(raw_path: str, allowed_folders: list[str]) -> Path:
    """Resolves ``raw_path`` and verifies it falls inside one of
    ``allowed_folders``. Raises ValueError otherwise.

    Resolves symlinks and ".." components before checking (Path.resolve())
    specifically so "allowed_folder/../../../etc/passwd" can't escape the
    sandbox by construction -- checking the raw, unresolved string against
    the allowed prefix would not catch that.
    """
    resolved = Path(raw_path).resolve()
    for folder in allowed_folders:
        folder_resolved = Path(folder).resolve()
        if resolved == folder_resolved or folder_resolved in resolved.parents:
            return resolved
    raise ValueError(
        f"{raw_path!r} is not inside any allowed folder. Allowed: {allowed_folders}"
    )


_APP_PARAM_KEYS = ("app", "app_name", "name", "application", "target", "url", "program")


class CapabilityDisabled(Exception):
    pass


class LocalAgent:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.capabilities: dict[str, bool] = config["capabilities"]
        self.allowed_folders: list[str] = config["allowed_folders"]
        self.app_aliases: dict[str, str] = config.get("app_aliases", {})

    def _require(self, capability: str) -> None:
        if not self.capabilities.get(capability, False):
            raise CapabilityDisabled(
                f"'{capability}' is disabled in this agent's local config.json."
            )

    # ---- open_apps ----

    def open_app(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("open_apps")
        # Small models don't reliably use the documented key, so accept
        # the names they actually produce.
        target = next(
            (params[k] for k in _APP_PARAM_KEYS if params.get(k)), None
        )
        if not target:
            raise ValueError("open_app requires 'app' (an app or website name).")
        return app_launcher.resolve_and_launch(str(target), self.app_aliases)

    def list_windows(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("open_apps")
        try:
            import pygetwindow as gw
        except ImportError as exc:
            raise RuntimeError(
                "list_windows requires the optional 'pygetwindow' package "
                "(pip install pygetwindow)."
            ) from exc
        return {"windows": [w.title for w in gw.getAllWindows() if w.title]}

    def focus_window(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("open_apps")
        title = params.get("title")
        if not title:
            raise ValueError("focus_window requires 'title'.")
        try:
            import pygetwindow as gw
        except ImportError as exc:
            raise RuntimeError(
                "focus_window requires the optional 'pygetwindow' package "
                "(pip install pygetwindow)."
            ) from exc
        matches = gw.getWindowsWithTitle(title)
        if not matches:
            raise ValueError(f"No window matching {title!r} found.")
        matches[0].activate()
        return {"focused": matches[0].title}

    def close_window(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("open_apps")
        title = params.get("title")
        if not title:
            raise ValueError("close_window requires 'title'.")
        try:
            import pygetwindow as gw
        except ImportError as exc:
            raise RuntimeError(
                "close_window requires the optional 'pygetwindow' package "
                "(pip install pygetwindow)."
            ) from exc
        matches = gw.getWindowsWithTitle(title)
        if not matches:
            raise ValueError(f"No window matching {title!r} found.")
        matches[0].close()
        return {"closed": matches[0].title}

    # ---- file_rw ----

    def read_file(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("file_rw")
        path = _resolve_allowed_path(params.get("path", ""), self.allowed_folders)
        return {"path": str(path), "content": path.read_text(encoding="utf-8")}

    def write_file(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("file_rw")
        path = _resolve_allowed_path(params.get("path", ""), self.allowed_folders)
        content = params.get("content", "")
        path.write_text(content, encoding="utf-8")
        return {"path": str(path), "bytes_written": len(content.encode("utf-8"))}

    def list_directory(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("file_rw")
        path = _resolve_allowed_path(params.get("path", ""), self.allowed_folders)
        return {"path": str(path), "entries": sorted(p.name for p in path.iterdir())}

    # ---- shell_exec ----

    def run_command(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("shell_exec")
        command = params.get("command")
        if not command:
            raise ValueError("run_command requires 'command'.")

        with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as audit:
            audit.write(f"{command}\n")

        result = subprocess.run(  # noqa: S602 -- intentional, gated by shell_exec
            command,
            shell=True,
            cwd=params.get("cwd"),
            capture_output=True,
            text=True,
            timeout=params.get("timeout_seconds", 30),
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }

    # ---- mouse_keyboard_control ----

    def _pyautogui(self):
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError(
                "mouse/keyboard actions require the optional 'pyautogui' package "
                "(pip install pyautogui)."
            ) from exc
        return pyautogui

    def move_mouse(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("mouse_keyboard_control")
        pyautogui = self._pyautogui()
        x, y = params.get("x"), params.get("y")
        if x is None or y is None:
            raise ValueError("move_mouse requires 'x' and 'y'.")
        pyautogui.moveTo(x, y)
        return {"x": x, "y": y}

    def click(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("mouse_keyboard_control")
        pyautogui = self._pyautogui()
        pyautogui.click(button=params.get("button", "left"))
        return {"button": params.get("button", "left")}

    def type_text(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("mouse_keyboard_control")
        pyautogui = self._pyautogui()
        text = params.get("text", "")
        pyautogui.typewrite(text)
        return {"characters_typed": len(text)}

    def press_key(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require("mouse_keyboard_control")
        pyautogui = self._pyautogui()
        key = params.get("key")
        if not key:
            raise ValueError("press_key requires 'key'.")
        pyautogui.press(key)
        return {"key": key}

    # ---- dispatch ----

    _HANDLERS = {
        "open_app": "open_app",
        "list_windows": "list_windows",
        "focus_window": "focus_window",
        "close_window": "close_window",
        "read_file": "read_file",
        "write_file": "write_file",
        "list_directory": "list_directory",
        "run_command": "run_command",
        "move_mouse": "move_mouse",
        "click": "click",
        "type_text": "type_text",
        "press_key": "press_key",
    }

    def handle_command(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        handler_name = self._HANDLERS.get(action)
        if handler_name is None:
            raise ValueError(f"Unknown action {action!r}.")
        handler = getattr(self, handler_name)
        return handler(params)


async def run(config_path: str) -> None:
    config = _load_config(config_path)
    agent = LocalAgent(config)

    server_url = config["server_url"]
    api_key = config["api_key"]
    url = f"{server_url}?api_key={api_key}"

    backoff = 1
    while True:
        try:
            async with websockets.connect(url) as ws:
                log.info("connected to %s", server_url)
                await ws.send(json.dumps({"type": "handshake", "capabilities": agent.capabilities}))
                backoff = 1  # reset after a successful connection

                async for raw_message in ws:
                    message = json.loads(raw_message)
                    if message.get("type") != "command":
                        continue

                    request_id = message.get("request_id")
                    action = message.get("action")
                    params = message.get("params", {})
                    log.info("received command: %s %s", action, params)

                    try:
                        details = agent.handle_command(action, params)
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "result",
                                    "request_id": request_id,
                                    "success": True,
                                    "details": details,
                                }
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("command failed: %s", exc)
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "result",
                                    "request_id": request_id,
                                    "success": False,
                                    "error": str(exc),
                                }
                            )
                        )
        except (websockets.ConnectionClosed, OSError) as exc:
            log.warning("disconnected (%s); reconnecting in %ss", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "config.json"
    )
    asyncio.run(run(config_file))
