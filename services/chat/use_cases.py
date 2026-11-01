"""Chat turn orchestration independent of FastAPI/WebSocket objects."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Protocol, cast

import asyncio
import structlog

from services.auth.principal import Principal
from services.chat.ports import HistoryStore
from services.chat.prompt import NO_CONTEXT_MESSAGE, build_messages
from services.llm_gateway.router import LLMGatewayError

log = structlog.get_logger()
SOURCE_PREVIEW_CHARS = 400

StreamCompletion = Callable[..., AsyncGenerator[str, None]]
RouteMessage = Callable[[str, str, Principal], Awaitable[dict[str, Any]]]
CondenseQuery = Callable[[list[dict[str, Any]], str, StreamCompletion], Awaitable[str]]
Retrieve = Callable[[str, str], Awaitable[list[dict[str, Any]]]]


class ChatSession(Protocol):
    principal: Principal

    async def receive_text(self) -> str: ...

    async def send_json(self, data: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class ChatUseCaseDependencies:
    history: HistoryStore
    route_message: RouteMessage
    stream_completion: Callable[[], StreamCompletion]
    condense_query: CondenseQuery
    retrieve: Retrieve


class ChatUseCase:
    """Run turns after the transport has authenticated and opened a session."""

    def __init__(self, dependencies: ChatUseCaseDependencies) -> None:
        self._dependencies = dependencies

    async def run(self, session: ChatSession, conversation_id: str) -> None:
        dependencies = self._dependencies
        principal = session.principal
        tenant_id = principal.tenant_id

        while True:
            user_query = await session.receive_text()
            log.info(
                "chat_message_received",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
            )

            try:
                history = (
                    await asyncio.to_thread(
                        dependencies.history.get_history, tenant_id, conversation_id
                    )
                    if "chat:history" in principal.scopes
                    else []
                )
                history = cast(list[dict[str, Any]], history)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "chat_history_read_failed",
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    error_type=type(exc).__name__,
                )
                await session.send_json(
                    {"type": "error", "detail": "Chat history is temporarily unavailable."}
                )
                continue

            route_state = await self._route_message(
                session, tenant_id, conversation_id, user_query, principal
            )
            if route_state is None:
                continue
            route = route_state["route"]

            if route == "tool":
                await self._complete_tool_turn(
                    session, tenant_id, conversation_id, user_query, route_state
                )
                continue

            if route == "conversational":
                await self._complete_conversational_turn(
                    session,
                    tenant_id,
                    conversation_id,
                    user_query,
                    history,
                )
                continue

            if "documents:read" not in principal.scopes:
                await session.send_json(
                    {"type": "error", "detail": "Document retrieval is not permitted."}
                )
                continue

            await self._complete_rag_turn(
                session,
                tenant_id,
                conversation_id,
                user_query,
                history,
            )

    async def _route_message(
        self,
        session: ChatSession,
        tenant_id: str,
        conversation_id: str,
        user_query: str,
        principal: Principal,
    ) -> dict[str, Any] | None:
        try:
            return await self._dependencies.route_message(tenant_id, user_query, principal)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent_routing_failed", tenant_id=tenant_id, error_type=type(exc).__name__)
            await session.send_json({"type": "error", "detail": "Could not process that message."})
            return None

    async def _complete_tool_turn(
        self,
        session: ChatSession,
        tenant_id: str,
        conversation_id: str,
        user_query: str,
        route_state: dict[str, Any],
    ) -> None:
        intent = route_state["intent"]
        tool_error = route_state.get("tool_error")
        if tool_error:
            await session.send_json({"type": "error", "detail": tool_error})
            return

        tool_result = route_state.get("tool_result")
        if tool_result is None:
            log.warning(
                "agent_tool_missing_result",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
            )
            await session.send_json(
                {"type": "error", "detail": "The requested action did not return a result."}
            )
            return

        saved = await self._persist_exchange(
            tenant_id, conversation_id, user_query, tool_result.summary
        )
        summary = tool_result.summary
        if not saved:
            summary += " Conversation saving is unconfirmed. The action may already have run; do not repeat it to repair chat history."
        await session.send_json(
            {
                "type": "tool_result",
                "tool": intent.tool_name,
                "action": intent.action,
                "success": tool_result.success,
                "summary": summary,
                "details": tool_result.details,
                "history_status": "stored" if saved else "unconfirmed",
            }
        )

    async def _complete_conversational_turn(
        self,
        session: ChatSession,
        tenant_id: str,
        conversation_id: str,
        user_query: str,
        history: list[dict[str, Any]],
    ) -> None:
        await session.send_json({"type": "sources", "sources": []})
        messages = build_messages(history, [], user_query)
        answer = await self._stream_answer(session, messages, tenant_id)
        if answer is not None:
            await self._complete_text_turn(
                session, tenant_id, conversation_id, user_query, answer, []
            )

    async def _complete_rag_turn(
        self,
        session: ChatSession,
        tenant_id: str,
        conversation_id: str,
        user_query: str,
        history: list[dict[str, Any]],
    ) -> None:
        stream_completion = self._dependencies.stream_completion()
        retrieval_query = await self._dependencies.condense_query(
            history, user_query, stream_completion
        )
        if retrieval_query != user_query:
            log.info(
                "query_condensed",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
            )

        try:
            sources = await self._dependencies.retrieve(retrieval_query, tenant_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("chat_retrieval_failed", tenant_id=tenant_id, error_type=type(exc).__name__)
            await session.send_json(
                {"type": "error", "detail": "Search is temporarily unavailable."}
            )
            return

        source_summaries = [
            {
                "index": index,
                "filename": source["filename"],
                "page_number": source["page_number"],
                "score": source.get("score"),
                "text": source["text"][:SOURCE_PREVIEW_CHARS],
            }
            for index, source in enumerate(sources, start=1)
        ]
        await session.send_json({"type": "sources", "sources": source_summaries})

        if not sources:
            await session.send_json({"type": "token", "content": NO_CONTEXT_MESSAGE})
            await self._complete_text_turn(
                session,
                tenant_id,
                conversation_id,
                user_query,
                NO_CONTEXT_MESSAGE,
                [],
            )
            return

        messages = build_messages(history, sources, user_query)
        answer = await self._stream_answer(session, messages, tenant_id, stream_completion)
        if answer is not None:
            await self._complete_text_turn(
                session,
                tenant_id,
                conversation_id,
                user_query,
                answer,
                source_summaries,
            )

    async def _stream_answer(
        self,
        session: ChatSession,
        messages: list[dict[str, Any]],
        tenant_id: str,
        stream_completion: StreamCompletion | None = None,
    ) -> str | None:
        stream_completion = stream_completion or self._dependencies.stream_completion()
        parts: list[str] = []
        try:
            async with aclosing(stream_completion(messages)) as stream:
                async for delta in stream:
                    parts.append(delta)
                    await session.send_json({"type": "token", "content": delta})
        except LLMGatewayError as exc:
            log.warning(
                "chat_generation_failed", tenant_id=tenant_id, error_type=type(exc).__name__
            )
            await session.send_json({"type": "error", "detail": str(exc)})
            return None
        return "".join(parts)

    async def _persist_exchange(
        self, tenant_id: str, conversation_id: str, user: str, assistant: str
    ) -> bool:
        try:
            await asyncio.to_thread(
                self._dependencies.history.append_exchange,
                tenant_id,
                conversation_id,
                user,
                assistant,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "chat_history_write_unconfirmed",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                error_type=type(exc).__name__,
            )
            return False
        return True

    async def _complete_text_turn(
        self,
        session: ChatSession,
        tenant_id: str,
        conversation_id: str,
        user: str,
        assistant: str,
        citations: list[dict[str, Any]],
    ) -> None:
        if await self._persist_exchange(tenant_id, conversation_id, user, assistant):
            await session.send_json({"type": "done", "citations": citations})
        else:
            await session.send_json(
                {
                    "type": "error",
                    "code": "history_write_unconfirmed",
                    "detail": "The response was produced, but saving this exchange could not be confirmed. It may already be stored; inspect conversation history before retrying.",
                }
            )
