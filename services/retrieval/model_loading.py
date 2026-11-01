"""Own process-wide model construction and bounded inference admission."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Condition, RLock
from typing import Iterator, Protocol


class InferenceSettings(Protocol):
    model_inference_concurrency: int
    model_inference_max_waiters: int
    model_inference_wait_timeout_seconds: float


class ModelInferenceRejected(RuntimeError):
    """Raised when the bounded model inference queue is full or times out."""


class ModelInferenceGate:
    """Bound simultaneous model calls and the number of waiting callers."""

    def __init__(
        self,
        concurrency: int = 1,
        max_waiters: int = 8,
        wait_timeout_seconds: float = 30,
    ) -> None:
        self._condition = Condition()
        self._active = 0
        self._waiting = 0
        self._rejected = 0
        self._configured = False
        if concurrency < 1 or max_waiters < 0 or wait_timeout_seconds <= 0:
            raise ValueError("Invalid model inference admission limits")
        self._limits = (concurrency, max_waiters, float(wait_timeout_seconds))

    def configure(self, concurrency: int, max_waiters: int, wait_timeout_seconds: float) -> None:
        limits = (concurrency, max_waiters, float(wait_timeout_seconds))
        if concurrency < 1 or max_waiters < 0 or wait_timeout_seconds <= 0:
            raise ValueError("Invalid model inference admission limits")
        with self._condition:
            if self._configured and limits == self._limits:
                return
            if self._configured:
                raise RuntimeError("Model inference limits must be consistent across one process")
            if self._active or self._waiting:
                raise RuntimeError("Cannot configure model inference while calls are active")
            self._limits = limits
            self._configured = True

    @property
    def limits(self) -> tuple[int, int, float]:
        with self._condition:
            return self._limits

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "rejected": self._rejected,
            }

    @contextmanager
    def inference(self) -> Iterator[None]:
        with self._condition:
            concurrency, max_waiters, timeout = self._limits
            if self._active >= concurrency:
                if self._waiting >= max_waiters:
                    self._rejected += 1
                    raise ModelInferenceRejected("Model inference capacity is full")
                self._waiting += 1
                try:
                    admitted = self._condition.wait_for(
                        lambda: self._active < concurrency, timeout=timeout
                    )
                finally:
                    self._waiting -= 1
                if not admitted:
                    self._rejected += 1
                    raise ModelInferenceRejected("Model inference admission timed out")
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()


MODEL_INFERENCE_GATE = ModelInferenceGate()


def configure_model_inference(settings: InferenceSettings) -> None:
    """Apply the process-wide inference limits from validated app settings."""
    MODEL_INFERENCE_GATE.configure(
        settings.model_inference_concurrency,
        settings.model_inference_max_waiters,
        settings.model_inference_wait_timeout_seconds,
    )


# Model construction is expensive and may allocate substantial CPU/GPU memory.
# lru_cache protects its mapping but may execute a concurrent cache miss more
# than once, so both embedding and reranker loads share this construction lock.
MODEL_LOAD_LOCK = RLock()
