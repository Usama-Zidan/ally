"""Device binding remains authoritative across socket and credential changes."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from services.agent.pc_agent_bridge import PCAgentBridge, PCAgentError
from services.auth.admission import AdmissionDenied, PairingAdmission
from services.auth.devices import DEVICE_SCOPES, DeviceRegistry, PairedDevice
from services.auth.principal import AuthorizationDenied, Principal
from services.auth.settings import SecuritySettings

DEVICE = PairedDevice("tenant-a", "credential-a", "550e8400-e29b-41d4-a716-446655440000")


class DeviceBindingTest(unittest.TestCase):
    def test_tenant_credentials_and_bad_device_ids_cannot_bind(self):
        registry = DeviceRegistry()
        registry.require_active = Mock()
        for scopes, device in (
            (frozenset({"pc:connect", "pc:execute"}), DEVICE.device_id),
            (DEVICE_SCOPES, "spoof"),
            (DEVICE_SCOPES, None),
        ):
            with self.subTest(scopes=scopes, device=device), self.assertRaises(AuthorizationDenied):
                registry.bind(Principal("tenant-a", "credential-a", scopes), device)
        registry.require_active.assert_not_called()

    def test_binding_requires_exact_credential_and_device(self):
        registry = DeviceRegistry()
        registry.require_active = Mock()
        principal = Principal(DEVICE.tenant_id, DEVICE.credential_id, DEVICE_SCOPES)
        self.assertEqual(registry.bind(principal, DEVICE.device_id), DEVICE)
        registry.require_active.assert_called_once_with(DEVICE)

    def test_pairing_admission_caps_hosts_attempts_and_concurrency(self):
        admission = PairingAdmission(
            SecuritySettings(
                max_auth_clients=1,
                auth_attempts_per_minute=2,
                max_auth_inflight=1,
            )
        )
        with admission.admit("a"):
            with self.assertRaises(AdmissionDenied):
                with admission.admit("a"):
                    self.fail("concurrency limit bypassed")
            with self.assertRaises(AdmissionDenied):
                with admission.admit("b"):
                    self.fail("host limit bypassed")
        with self.assertRaises(RuntimeError):
            with admission.admit("a"):
                raise RuntimeError("cancelled request")
        self.assertEqual(admission.inflight, 0)
        self.assertEqual(len(admission.attempts), 1)
        with self.assertRaises(AdmissionDenied):
            with admission.admit("a"):
                self.fail("rate limit bypassed")


class BridgeDeviceTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.check = AsyncMock()
        self.bridge = PCAgentBridge(check_device=self.check)
        self.ws = Mock(send_json=AsyncMock(), close=AsyncMock())

    async def register(self):
        await self.bridge.register(DEVICE.tenant_id, self.ws, {"open_apps": True}, device=DEVICE)

    async def test_required_binding_cannot_be_omitted_or_cross_tenant(self):
        for device in (None, PairedDevice("other", "key", DEVICE.device_id)):
            with self.assertRaises(ValueError):
                await self.bridge.register(DEVICE.tenant_id, self.ws, {}, device=device)
        self.check.assert_not_awaited()

    async def test_revoked_credential_cannot_deliver_a_command(self):
        await self.register()
        self.check.side_effect = AuthorizationDenied()
        with self.assertRaises(PCAgentError):
            await self.bridge.send_command(DEVICE.tenant_id, "open_apps", "open_app", {})
        self.ws.send_json.assert_not_awaited()
        self.assertFalse(self.bridge._connections[DEVICE.tenant_id].pending)

    async def test_revoked_credential_cannot_resolve_an_inflight_result(self):
        await self.register()
        sent = asyncio.Event()
        self.ws.send_json.side_effect = lambda _: sent.set()
        command = asyncio.create_task(
            self.bridge.send_command(
                DEVICE.tenant_id,
                "open_apps",
                "open_app",
                {},
                timeout=2,
            )
        )
        await sent.wait()
        message = self.ws.send_json.call_args.args[0]
        self.check.side_effect = AuthorizationDenied()
        await self.bridge.handle_result(
            DEVICE.tenant_id,
            {
                "request_id": message["request_id"],
                "success": True,
            },
            self.ws,
        )
        with self.assertRaises(PCAgentError):
            await command

    async def test_replacement_during_verification_cannot_publish_old_result(self):
        await self.register()
        sent = asyncio.Event()
        self.ws.send_json.side_effect = lambda _: sent.set()
        command = asyncio.create_task(
            self.bridge.send_command(
                DEVICE.tenant_id,
                "open_apps",
                "open_app",
                {},
                timeout=2,
            )
        )
        await sent.wait()
        message = self.ws.send_json.call_args.args[0]
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_check(_):
            entered.set()
            await release.wait()

        self.check.side_effect = delayed_check
        result = asyncio.create_task(
            self.bridge.handle_result(
                DEVICE.tenant_id,
                {
                    "request_id": message["request_id"],
                    "success": True,
                },
                self.ws,
            )
        )
        await entered.wait()
        self.check.side_effect = None
        await self.bridge.register(DEVICE.tenant_id, Mock(close=AsyncMock()), {}, device=DEVICE)
        release.set()
        await result
        with self.assertRaises(PCAgentError):
            await command
