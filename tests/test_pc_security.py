"""Regression tests for source identity, bounded commands and local consent."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from services.agent.pc_agent_bridge import PCAgentBridge, PCAgentError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_agent"))
from agent import LocalAgent  # noqa: E402
from shell_runner import ShellResult  # noqa: E402


def socket():
    return Mock(send_json=AsyncMock(), close=AsyncMock())


class ConnectionIdentityTest(unittest.IsolatedAsyncioTestCase):
    async def test_stale_disconnect_cannot_remove_replacement(self):
        bridge = PCAgentBridge()
        old, new = socket(), socket()
        await bridge.register("a", old, {"open_apps": True})
        await bridge.register("a", new, {"open_apps": True})
        bridge.unregister("a", old)
        self.assertTrue(bridge.is_connected("a"))
        bridge.unregister("a", new)
        self.assertFalse(bridge.is_connected("a"))

    async def test_stale_or_cross_tenant_result_cannot_resolve_new_command(self):
        bridge = PCAgentBridge()
        old, new, other = socket(), socket(), socket()
        await bridge.register("a", old, {"open_apps": True})
        await bridge.register("a", new, {"open_apps": True})
        await bridge.register("b", other, {"open_apps": True})
        task = asyncio.create_task(bridge.send_command("a", "open_apps", "open_app", {}))
        await asyncio.sleep(0)
        request_id = next(iter(bridge._connections["a"].pending))
        result = {"request_id": request_id, "success": True, "details": {"ok": True}}
        await bridge.handle_result("a", result, old)
        await bridge.handle_result("b", result, other)
        self.assertFalse(task.done())
        await bridge.handle_result("a", result, new)
        self.assertEqual(await task, {"ok": True})

    async def test_cancellation_releases_pending_slot(self):
        bridge = PCAgentBridge()
        ws = socket()
        await bridge.register("a", ws, {"open_apps": True})
        task = asyncio.create_task(bridge.send_command("a", "open_apps", "open_app", {}))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(bridge._connections["a"].pending, {})

    async def test_deadline_includes_blocked_socket_send(self):
        bridge = PCAgentBridge()
        ws = socket()
        ws.send_json.side_effect = lambda _: None

        async def blocked(_):
            await asyncio.Event().wait()

        ws.send_json.side_effect = blocked
        await bridge.register("a", ws, {"open_apps": True})
        with self.assertRaises(PCAgentError):
            await asyncio.wait_for(
                bridge.send_command("a", "open_apps", "open_app", {}, timeout=0.01),
                timeout=0.5,
            )
        self.assertEqual(bridge._connections["a"].pending, {})

    async def test_cancellation_during_send_releases_pending_slot(self):
        bridge = PCAgentBridge()
        ws = socket()
        entered = asyncio.Event()

        async def blocked(_):
            entered.set()
            await asyncio.Event().wait()

        ws.send_json.side_effect = blocked
        await bridge.register("a", ws, {"open_apps": True})
        task = asyncio.create_task(bridge.send_command("a", "open_apps", "open_app", {}))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(bridge._connections["a"].pending, {})

    async def test_pending_commands_are_bounded(self):
        bridge = PCAgentBridge(max_pending_commands=1)
        ws = socket()
        await bridge.register("a", ws, {"open_apps": True})
        task = asyncio.create_task(bridge.send_command("a", "open_apps", "open_app", {}))
        await asyncio.sleep(0)
        with self.assertRaises(PCAgentError):
            await bridge.send_command("a", "open_apps", "open_app", {})
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_capabilities_must_be_real_booleans(self):
        bridge = PCAgentBridge()
        with self.assertRaises(ValueError):
            await bridge.register("a", socket(), {"shell_exec": "false"})

    async def test_capabilities_are_copied_at_boundary(self):
        bridge = PCAgentBridge()
        capabilities = {"open_apps": False}
        await bridge.register("a", socket(), capabilities)
        capabilities["open_apps"] = True
        exposed = bridge.get_capabilities("a")
        exposed["open_apps"] = True
        self.assertEqual(bridge.get_capabilities("a"), {"open_apps": False})


class LocalApprovalTest(unittest.TestCase):
    def agent(self, approve=None, **capabilities):
        return LocalAgent({"capabilities": capabilities, "allowed_folders": []}, approve=approve)

    def test_unattended_shell_is_denied_without_subprocess(self):
        agent = self.agent(shell_exec=True)
        with (
            patch("agent.sys.stdin.isatty", return_value=False),
            patch("agent.run_shell") as run,
        ):
            with self.assertRaises(PermissionError):
                agent.handle_command("run_command", {"command": "echo never"})
        run.assert_not_called()

    def test_direct_write_handler_cannot_bypass_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "document.txt"
            agent = self.agent(approve=lambda action, params: False, file_rw=True)
            agent.allowed_folders = [tmp]
            with self.assertRaises(PermissionError):
                agent.write_file({"path": str(target), "content": "never"})
            self.assertFalse(target.exists())

    def test_forged_remote_approval_cannot_authorize_shell(self):
        agent = self.agent(approve=lambda action, params: False, shell_exec=True)
        with patch("agent.run_shell") as run:
            with self.assertRaises((PermissionError, ValueError)):
                agent.handle_command("run_command", {"command": "echo never", "confirmed": True})
        run.assert_not_called()

    def test_opening_script_requires_local_approval(self):
        agent = self.agent(approve=lambda action, params: False, open_apps=True)
        with patch("agent.app_launcher.resolve_and_launch") as launch:
            with self.assertRaises(PermissionError):
                agent.open_app({"app_name": "payload.cmd"})
        launch.assert_not_called()

    def test_approved_write_uses_exact_previewed_arguments(self):
        approve = Mock(return_value=True)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "document.txt"
            agent = self.agent(approve=approve, file_rw=True)
            agent.allowed_folders = [tmp]
            params = {"path": str(target), "content": "authorized"}
            agent.handle_command("write_file", params)
            self.assertEqual(target.read_text(), "authorized")
        approve.assert_called_once_with("write_file", params)

    def test_disabled_or_invalid_capability_cannot_be_approved(self):
        approve = Mock(return_value=True)
        agent = self.agent(approve=approve, shell_exec="false")
        with self.assertRaises(Exception):
            agent.run_command({"command": "echo never"})
        approve.assert_not_called()

    def test_shell_timeout_and_parameter_types_are_validated(self):
        approve = Mock(return_value=True)
        agent = self.agent(approve=approve, shell_exec=True)
        for params in (
            {"command": []},
            {"command": "echo x", "timeout_seconds": float("nan")},
            {"command": "echo x", "timeout_seconds": 1000000},
            {"command": "echo x", "timeout_seconds": True},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                agent.run_command(params)
        approve.assert_not_called()

    def test_approval_after_command_expiry_does_not_execute(self):
        agent = self.agent(approve=lambda action, params: True, open_apps=True)
        with (
            patch("agent.time.time", return_value=100),
            patch("agent.time.monotonic", side_effect=[10, 11, 41]),
            patch("agent.app_launcher.resolve_and_launch") as launch,
        ):
            with self.assertRaises(PermissionError):
                agent.handle_command("open_app", {"app_name": "notepad"}, expires_at=130)
        launch.assert_not_called()

    def test_expired_command_does_not_even_request_approval(self):
        approve = Mock(return_value=True)
        agent = self.agent(approve=approve, open_apps=True)
        with (
            patch("agent.time.time", return_value=100),
            patch("agent.time.monotonic", return_value=10),
        ):
            with self.assertRaises(PermissionError):
                agent.handle_command("open_app", {"app_name": "notepad"}, expires_at=99)
        approve.assert_not_called()

    def test_shell_audit_records_identity_without_raw_command_secrets(self):
        agent = self.agent(approve=lambda action, params: True, shell_exec=True)
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.log"
            with (
                patch("agent._AUDIT_LOG_PATH", audit_path),
                patch(
                    "agent.run_shell",
                    return_value=ShellResult("exited", returncode=0, started=True),
                ) as run,
            ):
                agent.run_command({"command": "secret-sentinel-argument"})
            events = [json.loads(line) for line in audit_path.read_text().splitlines()]
            self.assertNotIn("secret-sentinel-argument", audit_path.read_text())
            self.assertEqual(
                [event["outcome"] for event in events], ["approved_for_execution", "exited"]
            )
            self.assertEqual(events[0]["execution_id"], events[1]["execution_id"])
            self.assertNotEqual(events[0]["event_id"], events[1]["event_id"])
            self.assertEqual(events[1]["effect_outcome"], "not_verified")
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
