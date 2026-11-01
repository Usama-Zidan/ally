"""Transport wiring keeps native credentials and socket identity intact."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from fastapi import WebSocketDisconnect

from services.api import main
from services.auth.principal import Principal, AuthorizationDenied
from services.auth.devices import DEVICE_SCOPES, PairedDevice

DEVICE = PairedDevice("a", "test-credential", "550e8400-e29b-41d4-a716-446655440000")


def session(ws):
    return Mock(
        principal=Principal("a", "test-credential", DEVICE_SCOPES),
        receive_json=ws.receive_json,
        send_json=AsyncMock(),
    )


def websocket(messages):
    bridge = Mock(register=AsyncMock(), handle_result=AsyncMock(), unregister=Mock())
    return Mock(
        headers={"x-api-key": "native-header-key"},
        query_params={},
        accept=AsyncMock(),
        close=AsyncMock(),
        receive_json=AsyncMock(side_effect=messages),
        app=Mock(
            state=Mock(
                device_registry=Mock(bind=Mock(return_value=DEVICE)),
                pc_agent_bridge=bridge,
            )
        ),
    )


class PCWebSocketTransportTest(unittest.IsolatedAsyncioTestCase):
    async def test_result_is_bound_to_source_socket(self):
        result = {"type": "result", "request_id": "command-1", "success": True}
        ws = websocket(
            [
                {"type": "handshake", "capabilities": {"open_apps": True}},
                result,
                WebSocketDisconnect(),
            ]
        )
        bridge = ws.app.state.pc_agent_bridge
        await main._pc_agent_session(ws, session(ws))
        bridge.register.assert_awaited_once_with("a", ws, {"open_apps": True}, device=DEVICE)
        bridge.handle_result.assert_awaited_once_with("a", result, ws)
        bridge.unregister.assert_called_once_with("a", ws)

    async def test_invalid_handshake_does_not_register_connection(self):
        for message in ([], "handshake", {"type": "result"}):
            ws = websocket([message])
            bridge = ws.app.state.pc_agent_bridge
            with self.subTest(message=message):
                await main._pc_agent_session(ws, session(ws))
            bridge.register.assert_not_awaited()
            ws.close.assert_awaited_once()

    async def test_cancellation_while_registering_cleans_up_exact_socket(self):
        ws = websocket([{"type": "handshake", "capabilities": {}}])
        bridge = ws.app.state.pc_agent_bridge
        bridge.register.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await main._pc_agent_session(ws, session(ws))
        bridge.unregister.assert_called_once_with("a", ws)

    async def test_unpaired_or_spoofed_device_never_registers(self):
        ws = websocket([{"type": "handshake", "device_id": "spoof", "capabilities": {}}])
        ws.app.state.device_registry.bind.side_effect = AuthorizationDenied()
        bridge = ws.app.state.pc_agent_bridge
        await main._pc_agent_session(ws, session(ws))
        bridge.register.assert_not_awaited()
        ws.close.assert_awaited_once_with(
            code=1008, reason="A paired device credential is required"
        )


if __name__ == "__main__":
    unittest.main()
