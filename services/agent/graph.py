"""
The agent orchestration graph: classify -> route to {tool, rag,
conversational}.

Token-level LLM streaming deliberately happens OUTSIDE this graph, in
services.api.main.chat_ws, not inside a graph node. Getting raw
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

from typing import Any, Literal, Required, TypedDict, cast

import structlog
from langgraph.graph import END, StateGraph

from services.agent.classifier import Intent, get_classifier
from services.agent.commands import parse_command
from services.agent.registry import get_registry
from services.agent.tools.base import ToolError, ToolResult

log = structlog.get_logger()

AgentRoute = Literal["rag", "tool", "conversational"]


class AgentState(TypedDict, total=False):
    tenant_id: Required[str]
    query: Required[str]
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


async def classify_node(state: AgentState) -> dict[str, Any]:
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
        registry = get_registry()
        tools_block = registry.describe_for_classifier(state["tenant_id"])
        classifier = get_classifier()
        intent = await classifier.classify(state["query"], tools_block)

    return {"intent": intent, "route": intent.kind}


async def tool_node(state: AgentState) -> dict[str, Any]:
    if "intent" not in state:
        raise RuntimeError("Tool node requires an intent.")
    intent = state["intent"]
    registry = get_registry()
    tool = registry.get(intent.tool_name) if intent.tool_name else None

    if tool is None:
        return {"tool_error": f"Unknown tool {intent.tool_name!r}."}
    if intent.action is None:
        return {"tool_error": f"No action specified for tool {intent.tool_name!r}."}

    try:
        result = await tool.execute(state["tenant_id"], intent.action, intent.params)
        return {"tool_result": result}
    except ToolError as exc:
        return {"tool_error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        # A tool implementation bug (not an expected ToolError) must still
        # surface as a clean chat error, not an unhandled exception that
        # crashes the WebSocket turn.
        log.warning("tool_execution_unexpected_error", tool=intent.tool_name, error=str(exc))
        return {"tool_error": f"{intent.tool_name} failed unexpectedly: {exc}"}


def _route_after_classify(state: AgentState) -> str:
    if "route" not in state:
        raise RuntimeError("Classifier node did not set a route.")
    return state["route"]


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("tool", tool_node)

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


async def route_message(tenant_id: str, query: str) -> RoutedAgentState:
    """Runs the graph for one user message. Returns the final state:
    `state["route"]` is always one of "rag" | "tool" | "conversational";
    for "tool" exactly one of `tool_result` / `tool_error` is set.
    """
    graph = get_graph()
    result = await graph.ainvoke({"tenant_id": tenant_id, "query": query})
    if not isinstance(result, dict):
        raise TypeError("Agent graph returned a non-dictionary state.")
    if not all(
        isinstance(result.get(key), str)
        for key in ("tenant_id", "query", "route")
    ) or not isinstance(result.get("intent"), Intent):
        raise RuntimeError("Agent graph returned an incomplete state.")
    if result["route"] not in ("rag", "tool", "conversational"):
        raise RuntimeError("Agent graph returned an invalid route.")
    return cast(RoutedAgentState, result)
