"""
The agent orchestration graph: classify -> route to {tool, rag,
conversational}.

Token-level LLM streaming deliberately happens OUTSIDE this graph, in
services.chat.use_cases.ChatUseCase, not inside a graph node. Getting raw
token-by-token streaming out through LangGraph's event-streaming API
requires version-sensitive wiring this project has no way to verify
against a live install (no network access during development -- see the
project's established practice of flagging unverified integrations
rather than guessing). Routing and tool execution don't need streaming
at all (a tool call is one request/response), so the graph handles those
in full; for "rag" and "conversational" it just decides the route and
hands back to the caller, which reuses the exact, already-tested Phase 3
streaming path (retrieve_async -> build_messages -> stream_chat_completion)
for both -- "conversational" is that same path with an empty sources list.

This is also where an explicit slash command (services.agent.commands)
short-circuits the classifier entirely: the user's own stated intent
always wins over a model's guess.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Required, TypedDict, cast

import structlog
from langgraph.graph import END, StateGraph

from services.agent.classifier import Intent, IntentClassifier, get_classifier
from services.agent.commands import parse_command
from services.agent.local_pc_intent import infer_local_pc_intent, web_search_phrase
from services.agent.policy import validate_explicit_pc_command
from services.agent.registry import get_registry
from services.agent.tools.base import ToolError, ToolRegistry, ToolResult
from services.auth.principal import AuthorizationDenied, Principal, authorize_tool

log = structlog.get_logger()

AgentRoute = Literal["rag", "tool", "conversational"]


class AgentState(TypedDict, total=False):
    tenant_id: Required[str]
    query: Required[str]
    principal: Principal
    intent: Intent
    route: AgentRoute
    tool_result: ToolResult | None
    tool_error: str | None


class RoutedAgentState(TypedDict, total=False):
    tenant_id: Required[str]
    query: Required[str]
    intent: Required[Intent]
    route: Required[AgentRoute]
    tool_result: ToolResult | None
    tool_error: str | None


async def _classify_node(
    state: AgentState,
    classifier: IntentClassifier | None,
    registry: ToolRegistry | None,
) -> dict[str, Any]:
    parsed = parse_command(state["query"])
    if parsed is not None:
        intent = Intent(
            kind="tool",
            tool_name=parsed.tool_name,
            action=parsed.action,
            params=parsed.params,
            reasoning="explicit slash command",
        )
    else:
        active_registry = registry if registry is not None else get_registry()
        # Availability currently reads a synchronous PostgreSQL vault. So I'm Keeping it
        # off the event loop; absence of tool scopes avoids credential I/O entirely.
        principal = state.get("principal")
        if principal is not None and principal.scopes & {"tools:read", "tools:write", "pc:execute"}:
            tools_block = await asyncio.to_thread(active_registry.describe_for_principal, principal)
        else:
            tools_block = "(no tools are authorized for this credential)"
        active_classifier = classifier if classifier is not None else get_classifier()
        intent = await active_classifier.classify(state["query"], tools_block)
        local_intent = infer_local_pc_intent(state["query"])
        if local_intent is not None and (
            local_intent.kind == "conversational" or "- local_pc:" in tools_block
        ):
            intent = local_intent

    return {"intent": intent, "route": intent.kind}


async def _tool_node(state: AgentState, registry: ToolRegistry | None) -> dict[str, Any]:
    if "intent" not in state:
        raise RuntimeError("Tool node requires an intent.")
    intent = state["intent"]
    active_registry = registry if registry is not None else get_registry()
    tool = active_registry.get(intent.tool_name) if intent.tool_name else None

    if tool is None:
        return {"tool_error": f"Unknown tool {intent.tool_name!r}."}
    if intent.action is None:
        return {"tool_error": f"No action specified for tool {intent.tool_name!r}."}

    try:
        authorize_tool(state.get("principal"), state["tenant_id"], tool.name, intent.action)
        validate_explicit_pc_command(state["query"], tool.name, intent.action, intent.params)
        params = dict(intent.params)
        if (
            tool.name == "local_pc"
            and intent.action == "open_app"
            and parse_command(state["query"]) is None
        ):
            target = next(
                (
                    str(params[key])
                    for key in (
                        "app_name",
                        "app",
                        "name",
                        "application",
                        "target",
                        "url",
                        "program",
                    )
                    if params.get(key)
                ),
                "",
            )
            search_phrase = web_search_phrase(state["query"], target)
            if search_phrase:
                params["search_query"] = search_phrase
        result = await tool.execute(state["tenant_id"], intent.action, params)
        return {"tool_result": result}
    except (ToolError, AuthorizationDenied) as exc:
        return {"tool_error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        # A tool implementation bug (not an expected ToolError) must still
        # surface as a clean chat error, not an unhandled exception that
        # crashes the WebSocket turn.
        log.warning(
            "tool_execution_unexpected_error", tool=intent.tool_name, error_type=type(exc).__name__
        )
        return {"tool_error": "The requested tool failed unexpectedly. Please try again later."}


def _route_after_classify(state: AgentState) -> str:
    if "route" not in state:
        raise RuntimeError("Classifier node did not set a route.")
    return state["route"]


async def classify_node(state: AgentState) -> dict[str, Any]:
    """Compatibility node using the default process dependencies."""
    return await _classify_node(state, None, None)


async def tool_node(state: AgentState) -> dict[str, Any]:
    """Compatibility node using the default process registry."""
    return await _tool_node(state, None)


def build_graph(
    classifier: IntentClassifier | None = None,
    registry: ToolRegistry | None = None,
):
    async def classify_with_dependencies(state: AgentState) -> dict[str, Any]:
        return await _classify_node(state, classifier, registry)

    async def tool_with_dependencies(state: AgentState) -> dict[str, Any]:
        return await _tool_node(state, registry)

    workflow = StateGraph(AgentState)
    workflow.add_node("classify", classify_with_dependencies)
    workflow.add_node("tool", tool_with_dependencies)

    workflow.set_entry_point("classify")
    workflow.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"tool": "tool", "rag": END, "conversational": END},
    )
    workflow.add_edge("tool", END)

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def route_message(
    tenant_id: str,
    query: str,
    *,
    principal: Principal | None = None,
    graph=None,
) -> RoutedAgentState:
    """Runs the graph for one user message. Returns the final state:
    `state["route"]` is always one of "rag" | "tool" | "conversational";
    for "tool" exactly one of `tool_result` / `tool_error` is set.
    """
    graph = graph if graph is not None else get_graph()
    state: AgentState = {"tenant_id": tenant_id, "query": query}
    if principal is not None:
        if principal.tenant_id != tenant_id:
            raise AuthorizationDenied("Principal tenant does not match the requested tenant")
        state["principal"] = principal
    result = await graph.ainvoke(state)
    if not isinstance(result, dict):
        raise TypeError("Agent graph returned a non-dictionary state.")
    if not all(
        isinstance(result.get(key), str) for key in ("tenant_id", "query", "route")
    ) or not isinstance(result.get("intent"), Intent):
        raise RuntimeError("Agent graph returned an incomplete state.")
    if result["route"] not in ("rag", "tool", "conversational"):
        raise RuntimeError("Agent graph returned an invalid route.")
    return cast(RoutedAgentState, result)
