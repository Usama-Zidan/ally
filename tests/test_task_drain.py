"""Owner shutdown waits for blocking work even after its caller is cancelled."""

from __future__ import annotations

import asyncio
import threading
import unittest

from services.runtime.task_drain import DrainableTasks, WorkAdmissionClosed


class DrainableTasksTest(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_drains_threadpool_work_after_request_cancellation(self) -> None:
        tracker = DrainableTasks()
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocking_operation() -> None:
            entered.set()
            release.wait(timeout=3)
            finished.set()

        caller = asyncio.create_task(tracker.run(asyncio.to_thread(blocking_operation)))
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller

        self.assertEqual(tracker.active, 1)
        shutdown = asyncio.create_task(tracker.close())
        await asyncio.sleep(0)
        self.assertFalse(shutdown.done())

        release.set()
        await asyncio.wait_for(shutdown, timeout=1)
        self.assertTrue(finished.is_set())
        self.assertEqual(tracker.active, 0)
        self.assertTrue(tracker.closing)

    async def test_closed_owner_rejects_new_work(self) -> None:
        tracker = DrainableTasks()
        await tracker.close()

        with self.assertRaises(WorkAdmissionClosed):
            await tracker.run(asyncio.sleep(0))


if __name__ == "__main__":
    unittest.main()
