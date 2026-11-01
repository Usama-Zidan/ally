"""Socket close/cancellation reaches the local shell owner during execution."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

LOCAL = str(Path(__file__).resolve().parents[1] / "local_agent")
if LOCAL not in sys.path:
    sys.path.insert(0, LOCAL)
import agent  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "shell_child.py"


class CompanionCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_waiter_signals_and_joins_worker_without_detaching_it(self):
        entered = threading.Event()
        exited = threading.Event()

        def execute(*args, cancel, **kwargs):
            entered.set()
            if not cancel.wait(timeout=3):
                raise RuntimeError("Cancellation did not reach worker")
            exited.set()
            return {"outcome": "cancelled"}

        ws = Mock(wait_closed=asyncio.Event().wait)
        local = Mock(handle_command=execute)
        task = asyncio.create_task(
            agent._run_with_disconnect(ws, local, "run_command", {}, expires_at=time.time() + 10)
        )
        self.assertTrue(await asyncio.to_thread(entered.wait, 2))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(exited.is_set())

    @unittest.skipUnless(os.name == "nt", "requires actual Windows shell owner")
    async def test_real_socket_close_stops_execution_and_records_cancellation(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit_path = Path(temporary) / "audit.log"
            local = agent.LocalAgent(
                {"capabilities": {"shell_exec": True}, "allowed_folders": []},
                approve=lambda action, params: True,
            )
            command = subprocess.list2cmdline([sys.executable, str(FIXTURE), "fork", temporary])
            finished = asyncio.get_running_loop().create_future()

            async def handle(ws):
                try:
                    result = await agent._run_with_disconnect(
                        ws, local, "run_command", {"command": command}, expires_at=time.time() + 10
                    )
                    finished.set_result(result)
                except Exception as exc:
                    finished.set_exception(exc)

            with patch.object(agent, "_AUDIT_LOG_PATH", audit_path):
                async with serve(handle, "127.0.0.1", 0) as server:
                    port = server.sockets[0].getsockname()[1]
                    async with connect(f"ws://127.0.0.1:{port}") as client:
                        async with asyncio.timeout(5):
                            while not (Path(temporary) / "child.pid").exists():
                                await asyncio.sleep(0.01)
                        await client.close()
                        self.assertIsNone(await asyncio.wait_for(finished, timeout=4))
            events = [json.loads(line) for line in audit_path.read_text().splitlines()]
            self.assertEqual(
                [event["outcome"] for event in events], ["approved_for_execution", "cancelled"]
            )
            self.assertTrue(events[-1]["cleanup_confirmed"])
            self.assertEqual(events[-1]["effect_outcome"], "not_verified")
