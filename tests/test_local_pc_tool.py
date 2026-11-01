"""Tests for services.agent.tools.local_pc_tool."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from services.agent.pc_agent_bridge import PCAgentError
from services.agent.tools.base import ToolError
from services.agent.tools.local_pc_tool import LocalPCTool, _ACTION_CAPABILITY


class LocalPCToolTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = LocalPCTool()

    async def test_numeric_slash_command_arguments_are_normalized(self):
        with patch(
            "services.agent.tools.local_pc_tool.bridge.send_command",
            new=AsyncMock(return_value={}),
        ) as send:
            await self.tool.execute("tenant-a", "move_mouse", {"x": "1", "y": "2"})
            self.assertEqual(send.call_args.args[3], {"x": 1, "y": 2})
            await self.tool.execute(
                "tenant-a", "run_command", {"command": "dir", "timeout_seconds": "5"}
            )
            self.assertEqual(send.call_args.args[3]["timeout_seconds"], 5.0)

    async def test_invalid_numeric_arguments_never_reach_bridge(self):
        with patch(
            "services.agent.tools.local_pc_tool.bridge.send_command", new=AsyncMock()
        ) as send:
            for timeout in ("nan", "inf", "-1", True, []):
                with self.subTest(timeout=timeout), self.assertRaises(ToolError):
                    await self.tool.execute(
                        "tenant-a", "run_command", {"command": "dir", "timeout_seconds": timeout}
                    )
        send.assert_not_awaited()

    def test_is_available_reflects_bridge_connection_state(self):
        with patch("services.agent.tools.local_pc_tool.bridge.is_connected", return_value=True):
            self.assertTrue(self.tool.is_available("tenant-a"))
        with patch("services.agent.tools.local_pc_tool.bridge.is_connected", return_value=False):
            self.assertFalse(self.tool.is_available("tenant-a"))

    async def test_unknown_action_raises_tool_error_without_touching_the_bridge(self):
        with patch(
            "services.agent.tools.local_pc_tool.bridge.send_command", new=AsyncMock()
        ) as send_command:
            with self.assertRaises(ToolError):
                await self.tool.execute("tenant-a", "format_hard_drive", {})
        send_command.assert_not_awaited()

    async def test_execute_dispatches_through_the_bridge_with_the_right_capability(self):
        with patch(
            "services.agent.tools.local_pc_tool.bridge.send_command",
            new=AsyncMock(return_value={"launched": "notepad"}),
        ) as send_command:
            result = await self.tool.execute("tenant-a", "open_app", {"app_name": "notepad"})

        send_command.assert_awaited_once_with(
            "tenant-a", "open_apps", "open_app", {"app_name": "notepad"}
        )
        self.assertTrue(result.success)
        self.assertIn("notepad", result.summary)
        self.assertEqual(result.details, {"launched": "notepad"})

    async def test_pc_agent_error_is_translated_to_tool_error(self):
        with patch(
            "services.agent.tools.local_pc_tool.bridge.send_command",
            new=AsyncMock(side_effect=PCAgentError("No PC agent is currently connected.")),
        ):
            with self.assertRaises(ToolError) as ctx:
                await self.tool.execute("tenant-a", "open_app", {"app_name": "notepad"})
        self.assertIn("No PC agent is currently connected", str(ctx.exception))

    def test_action_descriptions_name_the_real_param_keys(self):
        # Regression: the classifier only ever sees these description
        # strings, and a vague one ("open an application by name") led
        # it to invent "app"/"url" instead of the real "app_name" in
        # production. Each description must name its actual required
        # param key(s) literally.
        expectations = {
            "open_app": ["app_name"],
            "focus_window": ["title"],
            "close_window": ["title"],
            "read_file": ["path"],
            "write_file": ["path", "content"],
            "list_directory": ["path"],
            "run_command": ["command"],
            "move_mouse": ["x", "y"],
            "type_text": ["text"],
            "press_key": ["key"],
        }
        for action, required_keys in expectations.items():
            description = self.tool.actions[action]
            for key in required_keys:
                self.assertIn(
                    key, description, f"{action}'s description doesn't name param {key!r}"
                )

    def test_run_command_description_specifies_windows_not_unix_syntax(self):
        # Regression: without this, the model defaulted to Unix syntax
        # (echo ... > /path/to/file.txt) for an agent that only ever
        # runs on the user's Windows PC.
        description = self.tool.actions["run_command"].lower()
        self.assertIn("windows", description)

    async def test_every_declared_action_maps_to_one_of_the_four_known_capabilities(self):
        known_capabilities = {"open_apps", "file_rw", "shell_exec", "mouse_keyboard_control"}
        for action in self.tool.actions:
            self.assertIn(action, _ACTION_CAPABILITY, f"{action} has no capability mapping")
            self.assertIn(_ACTION_CAPABILITY[action], known_capabilities)

    async def test_run_command_maps_to_shell_exec_specifically(self):
        with patch(
            "services.agent.tools.local_pc_tool.bridge.send_command",
            new=AsyncMock(return_value={"returncode": 0, "stdout": "", "stderr": ""}),
        ) as send_command:
            await self.tool.execute("tenant-a", "run_command", {"command": "dir"})

        self.assertEqual(send_command.call_args.args[1], "shell_exec")

    async def test_shell_success_requires_observed_exit_cleanup_and_audit(self):
        valid = {
            "outcome": "exited",
            "returncode": 0,
            "cleanup_confirmed": True,
            "audit_recorded": True,
        }
        for changes in (
            {},
            {"returncode": 7},
            {"outcome": "timed_out"},
            {"cleanup_confirmed": False},
            {"audit_recorded": False},
            {"returncode": False},
        ):
            details = {**valid, **changes}
            with patch(
                "services.agent.tools.local_pc_tool.bridge.send_command",
                new=AsyncMock(return_value=details),
            ):
                result = await self.tool.execute(
                    "tenant-a", "run_command", {"command": "private-command"}
                )
            self.assertEqual(result.success, not changes)
            self.assertNotIn("private-command", result.summary)
            self.assertIn(
                "unverified"
                if details["outcome"] == "exited" and type(details["returncode"]) is int
                else "partial",
                result.summary,
            )

    async def test_old_or_malformed_companion_cannot_claim_shell_success(self):
        for details in ({}, {"returncode": 0}, {"outcome": {"forged": "exited"}}):
            with patch(
                "services.agent.tools.local_pc_tool.bridge.send_command",
                new=AsyncMock(return_value=details),
            ):
                result = await self.tool.execute(
                    "tenant-a", "run_command", {"command": "private-command"}
                )
            self.assertFalse(result.success)

    async def test_mouse_and_keyboard_actions_map_to_mouse_keyboard_control(self):
        with patch(
            "services.agent.tools.local_pc_tool.bridge.send_command",
            new=AsyncMock(return_value={}),
        ) as send_command:
            for action, params in [
                ("move_mouse", {"x": 1, "y": 2}),
                ("click", {}),
                ("type_text", {"text": "hi"}),
                ("press_key", {"key": "enter"}),
            ]:
                send_command.reset_mock()
                await self.tool.execute("tenant-a", action, params)
                self.assertEqual(send_command.call_args.args[1], "mouse_keyboard_control")


if __name__ == "__main__":
    unittest.main()
