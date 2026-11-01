"""Bounded per-process admission for native device enrollment requests."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from typing import Iterator

from services.auth.settings import SecuritySettings


class AdmissionDenied(Exception):
    pass


class PairingAdmission:
    def __init__(self, settings: SecuritySettings) -> None:
        self.settings = settings
        self.attempts: OrderedDict[str, deque[float]] = OrderedDict()
        self.inflight = 0

    @contextmanager
    def admit(self, host: str) -> Iterator[None]:
        # Used on the owning event loop; there is no await before reservation.
        now = time.monotonic()
        while self.attempts and next(iter(self.attempts.values()))[-1] <= now - 60:
            self.attempts.popitem(last=False)
        attempts = self.attempts.get(host)
        if attempts is None:
            if len(self.attempts) >= self.settings.max_auth_clients:
                raise AdmissionDenied()
            attempts = self.attempts[host] = deque()
        while attempts and attempts[0] <= now - 60:
            attempts.popleft()
        if (
            len(attempts) >= self.settings.auth_attempts_per_minute
            or self.inflight >= self.settings.max_auth_inflight
        ):
            # Do not retain an empty deque when global capacity is full.
            if not attempts:
                del self.attempts[host]
            raise AdmissionDenied()
        attempts.append(now)
        self.attempts.move_to_end(host)
        self.inflight += 1
        try:
            yield
        finally:
            self.inflight -= 1
