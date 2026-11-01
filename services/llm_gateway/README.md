# LLM gateway (Phase 3)

`router.py` wraps a `litellm.Router` with a `pybreaker` circuit breaker:

- **Primary**: self-hosted Qwen3 via vLLM's OpenAI-compatible server
  (`QWEN_VLLM_BASE_URL`, `QWEN_MODEL_NAME`).
- **Fallback**: `LLM_FALLBACK_MODEL` (default `gpt-4o-mini`) via OpenAI,
  only registered when `OPENAI_API_KEY` is set. Router retries the
  primary `LLM_MAX_RETRIES` times before failing over.
- **Circuit breaker**: opens after `CIRCUIT_BREAKER_FAIL_MAX` consecutive
  total failures (every retry + fallback exhausted) and short-circuits
  new calls for `CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS`.

Public API: `async for delta in stream_chat_completion(messages): ...`,
raising `LLMGatewayError` (never a raw litellm/pybreaker exception) on
any failure — including a failure partway through an already-started
stream. Called from `services.api.main.chat_ws`; see the top-level
README's "Chat: LLM gateway + streaming with citations" section for the
full request flow.

The API composition root injects an `LLMGateway` built from each app's
immutable settings into chat streaming, query condensation and intent
classification. Its Router and breaker are app-scoped. LiteLLM's HTTP client
cache is process-scoped and cleaned up by LiteLLM at interpreter exit; Router
has no supported close method, and its reset operation mutates shared global
state, so app shutdown must not call either operation on shared LiteLLM state.
The module-level functions remain as compatibility entrypoints for standalone
callers.
