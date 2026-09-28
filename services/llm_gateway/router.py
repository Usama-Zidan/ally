"""
LLM gateway: a single streaming entrypoint in front of the self-hosted
Qwen3 deployment (via vLLM's OpenAI-compatible server) with automatic
fallback to a hosted provider.

Built on litellm.Router rather than a bare litellm.acompletion() call
because Router is what actually gives this "production-grade" retry and
fallback behavior: num_retries retries a failing model before moving on,
and the fallbacks list defines which model to try next. Re-implementing
that with a manual try/except per model would just be a worse version of
what Router already does.

A pybreaker CircuitBreaker sits in front of Router as a second layer:
Router's retries+fallbacks bound the latency of *one* request, but they
don't stop the gateway from paying that full retry+fallback cost again on
the *next* request if the backend is simply down. The breaker opens after
CIRCUIT_BREAKER_FAIL_MAX consecutive total failures and short-circuits
new calls for CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS, so a dead backend
fails fast instead of making every chat message wait out a full
retry+fallback cycle only to fail anyway.

Note: pybreaker's async entrypoint (call_async) is implemented on top of
Tornado's `gen.coroutine`, not asyncio, so `tornado` is a hard requirement
of this module rather than an optional extra (it is pinned in
requirements.txt for exactly this reason). _get_breaker() fails fast with
an actionable message if it is missing, because pybreaker's own failure in
that case is a bare `NameError: name 'gen' is not defined` raised from
inside the breaker on every single chat message.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, cast

import litellm
import pybreaker
import structlog
from litellm.router import Router

from config import (
    CIRCUIT_BREAKER_FAIL_MAX,
    CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS,
    LLM_FALLBACK_MODEL,
    LLM_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    QWEN_MODEL_NAME,
    QWEN_VLLM_BASE_URL,
)

log = structlog.get_logger()

# litellm's own retry/exception-handling logging is noisy at info level
# and duplicates what this module already logs around the breaker.
litellm.suppress_debug_info = True

PRIMARY_MODEL = QWEN_MODEL_NAME if QWEN_MODEL_NAME else "qwen-primary"
FALLBACK_MODEL = "openai-fallback"


class LLMGatewayError(Exception):
    """Raised when the gateway cannot produce a response.

    Covers both Router exhausting every model (primary + fallbacks) and
    the circuit breaker rejecting a call while open. Callers (the chat
    WebSocket handler) catch this one type rather than needing to know
    about litellm's or pybreaker's specific exception classes.
    """


def _build_model_list() -> list[dict]:
    """Builds Router's model_list: the self-hosted Qwen3 deployment is
    always included; the hosted fallback is only added when an API key is
    actually configured, so a dev environment without OPENAI_API_KEY set
    doesn't silently depend on it and get a confusing auth error mid-chat.
    """
    model_list = [
        {
            "model_name": PRIMARY_MODEL,
            "litellm_params": {
                # vLLM serves an OpenAI-compatible API, so "openai" is the
                # correct litellm provider even though this isn't OpenAI.
                "model": f"openai/{QWEN_MODEL_NAME}",
                "api_base": QWEN_VLLM_BASE_URL,
                "api_key": "not-needed",  # vLLM's OpenAI-compatible server doesn't check this
                "timeout": LLM_REQUEST_TIMEOUT_SECONDS,
            },
        }
    ]
    if OPENAI_API_KEY:
        model_list.append(
            {
                "model_name": FALLBACK_MODEL,
                "litellm_params": {
                    "model": LLM_FALLBACK_MODEL,
                    "api_key": OPENAI_API_KEY,
                    "timeout": LLM_REQUEST_TIMEOUT_SECONDS,
                },
            }
        )
    return model_list


def _build_router() -> Router:
    model_list = _build_model_list()
    fallbacks = [{PRIMARY_MODEL: [FALLBACK_MODEL]}] if OPENAI_API_KEY else []
    return Router(
        model_list=model_list,
        fallbacks=fallbacks,
        num_retries=LLM_MAX_RETRIES,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )


_router: Router | None = None
_breaker: pybreaker.CircuitBreaker | None = None


def _get_router() -> Router:
    global _router
    if _router is None:
        _router = _build_router()
        log.info(
            "llm_router_initialized",
            primary_model=QWEN_MODEL_NAME,
            fallback_enabled=bool(OPENAI_API_KEY),
        )
    return _router


def _get_breaker() -> pybreaker.CircuitBreaker:
    global _breaker
    if not pybreaker.HAS_TORNADO_SUPPORT:
        raise ImportError(
            "tornado is required by pybreaker's async circuit breaker "
            "(services/llm_gateway/router.py) — install requirements.txt"
        )
    if _breaker is None:
        _breaker = pybreaker.CircuitBreaker(
            fail_max=CIRCUIT_BREAKER_FAIL_MAX,
            reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS,
        )
    return _breaker


async def _start_stream(messages: list[dict], temperature: float | None = .7, max_tokens: int | None = 1024) -> AsyncIterator[Any]:
    """The call the breaker guards: asks Router for a streaming
    completion. Router itself has already retried/failed-over across
    every configured model by the time this either returns a stream or
    raises — this coroutine is what "one attempt" means to the breaker.
    """
    params: dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    router = _get_router()
    return await router.acompletion(
        model=PRIMARY_MODEL,
        messages= cast(Any,messages),
        stream=True,
        **params,
    )


async def stream_chat_completion(messages: list[dict], temperature: float | None = 0.7, max_tokens: int | None = 1024) -> AsyncIterator[str]:
    """Streams the assistant's reply as a sequence of text deltas.

    Args:
        messages: Chat-format messages (system/user/assistant dicts) —
            see services.chat.prompt.build_messages.

    Yields:
        Successive text fragments as the model generates them.

    Raises:
        LLMGatewayError: If the circuit breaker is open, or every
            configured model (primary and fallback) failed for this
            request.
    """
    breaker = _get_breaker()
    try:
        # pybreaker's call_async is a Tornado coroutine (see the module
        # docstring) whose runtime contract is a plain awaitable;
        # pybreaker's type stubs describe it as returning a generator,
        # hence the cast.
        call_async = cast(Any, breaker.call_async)
        stream = await call_async(
            _start_stream,
            messages, 
            temperature=temperature,
            max_tokens=max_tokens,
            )
    except pybreaker.CircuitBreakerError as exc:
        log.warning("llm_gateway_circuit_open")
        raise LLMGatewayError(
            "The chat model is temporarily unavailable after repeated failures; "
            "please try again shortly."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.warning("llm_gateway_request_failed", error=str(exc))
        raise LLMGatewayError(f"Chat model request failed: {exc}") from exc

    try:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:  # noqa: BLE001
        # A failure mid-stream (after the breaker already recorded this
        # call as a success for having started) still needs to reach the
        # caller as a gateway error, not a raw litellm/httpx exception.
        log.warning("llm_gateway_stream_interrupted", error=str(exc))
        raise LLMGatewayError(f"Chat model stream interrupted: {exc}") from exc


def reset_gateway_state() -> None:
    """Clears the cached Router and breaker so a config change (or a test)
    takes effect on the next call instead of reusing a stale instance."""
    global _router, _breaker
    _router = None
    _breaker = None
