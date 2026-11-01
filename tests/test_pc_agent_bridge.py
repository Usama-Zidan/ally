"""Tests for services.agent.pc_agent_bridge."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from services.agent.pc_agent_bridge import PCAgentBridge, PCAgentError


def _fake_websocket():
    ws = MagicMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


class RegistrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_close_fails_pending_calls_closes_socket_and_rejects_new_work(self):
        bridge = PCAgentBridge()
        ws = _fake_websocket()
        await bridge.register("tenant-a", ws, {"open_apps": True})
        send_task = asyncio.create_task(
            bridge.send_command("tenant-a", "open_apps", "open_app", {}, timeout=5)
        )
        await asyncio.sleep(0)

        await bridge.close()
        with self.assertRaisesRegex(PCAgentError, "shutting down"):
            await send_task
        ws.close.assert_awaited_once_with(code=1012, reason="Service shutting down")
        self.assertFalse(bridge.is_connected("tenant-a"))

        with self.assertRaisesRegex(PCAgentError, "shutting down"):
            await bridge.send_command("tenant-a", "open_apps", "open_app", {})
        with self.assertRaisesRegex(PCAgentError, "shutting down"):
            await bridge.register("tenant-b", _fake_websocket(), {})

    async def test_close_is_idempotent(self):
        bridge = PCAgentBridge()
        ws = _fake_websocket()
        await bridge.register("tenant-a", ws, {})
        await bridge.close()
        await bridge.close()
        ws.close.assert_awaited_once()

    async def test_is_connected_reflects_registration_state(self):
        bridge = PCAgentBridge()
        self.assertFalse(bridge.is_connected("tenant-a"))
        ws = _fake_websocket()
        await bridge.register("tenant-a", ws, {"open_apps": True})
        self.assertTrue(bridge.is_connected("tenant-a"))
        bridge.unregister("tenant-a", ws)
        self.assertFalse(bridge.is_connected("tenant-a"))

    async def test_get_capabilities_returns_none_when_not_connected(self):
        bridge = PCAgentBridge()
        self.assertIsNone(bridge.get_capabilities("tenant-a"))

    async def test_get_capabilities_returns_what_was_registered(self):
        bridge = PCAgentBridge()
        await bridge.register("tenant-a", _fake_websocket(), {"shell_exec": True})
        self.assertEqual(bridge.get_capabilities("tenant-a"), {"shell_exec": True})

    async def test_a_new_connection_replaces_the_previous_one_for_the_same_tenant(self):
        bridge = PCAgentBridge()
        first_ws = _fake_websocket()
        await bridge.register("tenant-a", first_ws, {"open_apps": True})
        second_ws = _fake_websocket()
        await bridge.register("tenant-a", second_ws, {"open_apps": False})

        self.assertEqual(bridge.get_capabilities("tenant-a"), {"open_apps": False})

    async def test_replacing_a_connection_fails_any_commands_still_pending_on_the_old_one(self):
        bridge = PCAgentBridge()
        first_ws = _fake_websocket()
        await bridge.register("tenant-a", first_ws, {"open_apps": True})

        send_task = asyncio.ensure_future(
            bridge.send_command("tenant-a", "open_apps", "open_app", {}, timeout=5)
        )
        await asyncio.sleep(0)  # let send_command register its pending future

        await bridge.register("tenant-a", _fake_websocket(), {"open_apps": True})

        with self.assertRaises(PCAgentError):
            await send_task

    async def test_unregister_fails_any_pending_commands(self):
        bridge = PCAgentBridge()
        ws = _fake_websocket()
        await bridge.register("tenant-a", ws, {"open_apps": True})

        send_task = asyncio.ensure_future(
            bridge.send_command("tenant-a", "open_apps", "open_app", {}, timeout=5)
        )
        await asyncio.sleep(0)

        bridge.unregister("tenant-a", ws)

        with self.assertRaises(PCAgentError):
            await send_task


class SendCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_raises_when_no_agent_is_connected(self):
        bridge = PCAgentBridge()
        with self.assertRaises(PCAgentError):
            await bridge.send_command("tenant-a", "open_apps", "open_app", {})

    async def test_raises_when_the_capability_is_not_enabled(self):
        bridge = PCAgentBridge()
        await bridge.register("tenant-a", _fake_websocket(), {"open_apps": False})

        with self.assertRaises(PCAgentError):
            await bridge.send_command("tenant-a", "open_apps", "open_app", {})

    async def test_sends_a_well_formed_command_message(self):
        bridge = PCAgentBridge()
        ws = _fake_websocket()
        await bridge.register("tenant-a", ws, {"open_apps": True})

        async def respond_after_send():
            await asyncio.sleep(0)
            sent = ws.send_json.call_args.args[0]
            await bridge.handle_result(
                "tenant-a",
                {
                    "type": "result",
                    "request_id": sent["request_id"],
                    "success": True,
                    "details": {"launched": "notepad"},
                },
                ws,
            )

        responder = asyncio.ensure_future(respond_after_send())
        details = await bridge.send_command(
            "tenant-a", "open_apps", "open_app", {"app_name": "notepad"}
        )
        await responder

        sent = ws.send_json.call_args.args[0]
        self.assertEqual(sent["type"], "command")
        self.assertEqual(sent["capability"], "open_apps")
        self.assertEqual(sent["action"], "open_app")
        self.assertEqual(sent["params"], {"app_name": "notepad"})
        self.assertEqual(details, {"launched": "notepad"})

    async def test_raises_when_the_agent_reports_failure(self):
        bridge = PCAgentBridge()
        ws = _fake_websocket()
        await bridge.register("tenant-a", ws, {"shell_exec": True})

        async def respond_with_failure():
            await asyncio.sleep(0)
            sent = ws.send_json.call_args.args[0]
            await bridge.handle_result(
                "tenant-a",
                {
                    "type": "result",
                    "request_id": sent["request_id"],
                    "success": False,
                    "error": "command not found",
                },
                ws,
            )

        responder = asyncio.ensure_future(respond_with_failure())
        with self.assertRaises(PCAgentError) as ctx:
            await bridge.send_command("tenant-a", "shell_exec", "run_command", {"command": "x"})
        await responder

        self.assertIn("declined or failed", str(ctx.exception))

    async def test_times_out_when_no_response_arrives(self):
        bridge = PCAgentBridge()
        await bridge.register("tenant-a", _fake_websocket(), {"open_apps": True})

        with self.assertRaises(PCAgentError) as ctx:
            await bridge.send_command("tenant-a", "open_apps", "open_app", {}, timeout=0.05)
        self.assertIn("did not respond", str(ctx.exception))

    async def test_result_for_an_unknown_request_id_is_ignored_not_an_error(self):
        # A stray/late result (e.g. after a timeout already resolved the
        # future) must not raise -- it's simply discarded.
        bridge = PCAgentBridge()
        ws = _fake_websocket()
        await bridge.register("tenant-a", ws, {"open_apps": True})
        await bridge.handle_result(
            "tenant-a", {"type": "result", "request_id": "does-not-exist", "success": True}, ws
        )  # must not raise

    async def test_result_without_a_string_request_id_is_ignored(self):
        bridge = PCAgentBridge()
        ws = _fake_websocket()
        await bridge.register("tenant-a", ws, {"open_apps": True})

        for request_id in (None, [], 123):
            await bridge.handle_result(
                "tenant-a", {"type": "result", "request_id": request_id, "success": True}, ws
            )

    async def test_result_for_an_unregistered_tenant_is_ignored_not_an_error(self):
        bridge = PCAgentBridge()
        await bridge.handle_result(
            "tenant-never-connected", {"request_id": "x", "success": True}, _fake_websocket()
        )  # must not raise

    async def test_send_failure_cleans_up_the_pending_future(self):
        bridge = PCAgentBridge()
        ws = _fake_websocket()
        ws.send_json.side_effect = RuntimeError("socket closed")
        await bridge.register("tenant-a", ws, {"open_apps": True})

        with self.assertRaises(PCAgentError):
            await bridge.send_command("tenant-a", "open_apps", "open_app", {})

        # No leaked pending future for a stray late result to resolve against.
        connection = bridge._connections["tenant-a"]
        self.assertEqual(connection.pending, {})


if __name__ == "__main__":
    unittest.main()
