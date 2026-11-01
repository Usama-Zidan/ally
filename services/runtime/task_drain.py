"""Track non-cancellable async/threadpool work until its owner can close."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import Any, TypeVar

T = TypeVar("T")


class WorkAdmissionClosed(RuntimeError):
    """Raised when an owner is shutting down and no longer accepts work."""


class DrainableTasks:
    """Own child tasks that must outlive cancellation of an awaiting request.

    The tracker is event-loop-local. Its owner calls ``close`` before closing
    any dependency used by its tasks.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closing = False

    @property
    def active(self) -> int:
        return len(self._tasks)

    @property
    def closing(self) -> bool:
        return self._closing

    async def run(self, awaitable: Awaitable[T]) -> T:
        task = self.start(awaitable)
        return await asyncio.shield(task)

    def start(self, awaitable: Awaitable[T]) -> asyncio.Task[T]:
        """Start and track work, returning its task for resource cleanup callbacks."""
        if self._closing:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise WorkAdmissionClosed("Work owner is shutting down")

        async def wait_for_work() -> T:
            return await awaitable

        task: asyncio.Task[T] = asyncio.create_task(wait_for_work())
        self._tasks.add(task)
        task.add_done_callback(self._discard)
        return task

    async def close(self) -> None:
        """Stop admission and wait until all owned work has completed."""
        self._closing = True
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def _discard(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            # Retrieve exceptions when the original waiter was cancelled.
            task.exception()
