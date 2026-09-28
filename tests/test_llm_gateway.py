"""
Tests for services.llm_gateway.router.

Most breaker behavior is tested with a small fake for determinism; a focused
test also exercises pybreaker's real async API to catch dependency/setup issues.
"""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from services.llm_gateway import router


class FakeCircuitBreakerError(Exception):
    pass


class FakeCircuitBreaker:
    """Reproduces pybreaker's documented behavior closely enough to test
    router.py's usage of it: opens after fail_max consecutive failures,
    stays open until reset_timeout elapses, and call_async(fn, *a, **kw)
    either returns fn's result, re-raises fn's exception (counted as a
    failure), or raises CircuitBreakerError while open (not counted)."""

    def __init__(self, fail_max: int, reset_timeout: int):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._consecutive_failures = 0
        self._open = False

    async def call_async(self, fn, *args, **kwargs):
        if self._open:
            raise FakeCircuitBreakerError("circuit is open")
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.fail_max:
                self._open = True
            raise
        else:
            self._consecutive_failures = 0
            return result


def install_fake_breaker(fail_max=3, reset_timeout=30):
    fake = FakeCircuitBreaker(fail_max, reset_timeout)
    return fake


async def _aiter(items):
    for item in items:
        yield item


def _make_chunk(text):
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=text))]
    return chunk


class BuildModelListTest(unittest.TestCase):
    def test_only_primary_model_when_no_fallback_key_configured(self):
        with patch.object(router, "OPENAI_API_KEY", ""):
            model_list = router._build_model_list()

        self.assertEqual(len(model_list), 1)
        self.assertEqual(model_list[0]["model_name"], router.PRIMARY_MODEL)
        self.assertTrue(model_list[0]["litellm_params"]["model"].startswith("openai/"))

    def test_fallback_model_added_when_key_is_configured(self):
        with patch.object(router, "OPENAI_API_KEY", "sk-test"):
            model_list = router._build_model_list()

        model_names = [m["model_name"] for m in model_list]
        self.assertEqual(model_names, [router.PRIMARY_MODEL, router.FALLBACK_MODEL])

    def test_build_router_only_configures_fallback_chain_when_key_present(self):
        with patch.object(router, "OPENAI_API_KEY", ""), patch.object(
            router, "Router"
        ) as router_cls:
            router._build_router()
        self.assertEqual(router_cls.call_args.kwargs["fallbacks"], [])

        with patch.object(router, "OPENAI_API_KEY", "sk-test"), patch.object(
            router, "Router"
        ) as router_cls:
            router._build_router()
        self.assertEqual(
            router_cls.call_args.kwargs["fallbacks"],
            [{router.PRIMARY_MODEL: [router.FALLBACK_MODEL]}],
        )


class StreamChatCompletionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        router.reset_gateway_state()
        self.addCleanup(router.reset_gateway_state)

    async def test_yields_only_non_empty_text_deltas(self):
        fake_router = MagicMock()
        fake_router.acompletion = AsyncMock(
            return_value=_aiter([_make_chunk("Hello "), _make_chunk(None), _make_chunk("world")])
        )
        with patch.object(router, "_get_router", return_value=fake_router), patch.object(
            router, "_get_breaker", return_value=install_fake_breaker()
        ):
            deltas = [d async for d in router.stream_chat_completion([{"role": "user", "content": "hi"}])]

        self.assertEqual(deltas, ["Hello ", "world"])
        fake_router.acompletion.assert_awaited_once()
        self.assertEqual(fake_router.acompletion.call_args.kwargs["model"], router.PRIMARY_MODEL)
        self.assertTrue(fake_router.acompletion.call_args.kwargs["stream"])

    async def test_real_breaker_supports_async_stream_calls(self):
        fake_router = MagicMock()
        fake_router.acompletion = AsyncMock(return_value=_aiter([_make_chunk("Hello")]))
        breaker = router.pybreaker.CircuitBreaker(fail_max=3, reset_timeout=30)

        with patch.object(router, "_get_router", return_value=fake_router), patch.object(
            router, "_get_breaker", return_value=breaker
        ):
            deltas = [
                delta
                async for delta in router.stream_chat_completion(
                    [{"role": "user", "content": "hi"}]
                )
            ]

        self.assertEqual(deltas, ["Hello"])

    async def test_missing_tornado_fails_fast_with_actionable_error(self):
        """pybreaker's async breaker needs Tornado. Without it, pybreaker
        raises a bare NameError from inside call_async on every message, so
        the gateway checks the dependency up front instead."""
        with patch.object(router.pybreaker, "HAS_TORNADO_SUPPORT", False):
            with self.assertRaises(ImportError) as ctx:
                async for _ in router.stream_chat_completion([{"role": "user", "content": "hi"}]):
                    pass

        self.assertIn("tornado", str(ctx.exception))

    async def test_initial_request_failure_raises_gateway_error(self):
        fake_router = MagicMock()
        fake_router.acompletion = AsyncMock(side_effect=RuntimeError("connection refused"))
        with patch.object(router, "_get_router", return_value=fake_router), patch.object(
            router, "_get_breaker", return_value=install_fake_breaker(fail_max=5)
        ):
            with self.assertRaises(router.LLMGatewayError):
                async for _ in router.stream_chat_completion([{"role": "user", "content": "hi"}]):
                    pass

    async def test_mid_stream_failure_raises_gateway_error(self):
        async def broken_stream():
            yield _make_chunk("partial ")
            raise RuntimeError("connection dropped mid-stream")

        fake_router = MagicMock()
        fake_router.acompletion = AsyncMock(return_value=broken_stream())
        with patch.object(router, "_get_router", return_value=fake_router), patch.object(
            router, "_get_breaker", return_value=install_fake_breaker()
        ):
            collected = []
            with self.assertRaises(router.LLMGatewayError):
                async for delta in router.stream_chat_completion([{"role": "user", "content": "hi"}]):
                    collected.append(delta)

        # Tokens received before the mid-stream failure must not be lost.
        self.assertEqual(collected, ["partial "])

    async def test_circuit_opens_after_consecutive_failures_and_short_circuits(self):
        fake_router = MagicMock()
        fake_router.acompletion = AsyncMock(side_effect=RuntimeError("backend down"))
        breaker = install_fake_breaker(fail_max=2)

        with patch.object(router, "_get_router", return_value=fake_router), patch.object(
            router, "_get_breaker", return_value=breaker
        ):
            for _ in range(2):
                with self.assertRaises(router.LLMGatewayError):
                    async for _ in router.stream_chat_completion([{"role": "user", "content": "hi"}]):
                        pass

            # Breaker is now open: the next call must fail fast without
            # Router being invoked a third time.
            with self.assertRaises(router.LLMGatewayError):
                async for _ in router.stream_chat_completion([{"role": "user", "content": "hi"}]):
                    pass

        self.assertEqual(fake_router.acompletion.await_count, 2)


if __name__ == "__main__":
    unittest.main()
