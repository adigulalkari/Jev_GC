"""LangGraph integration (SPEC.md §4.9).

`jev_gc_node` builds a LangGraph node that trims a graph's message history
through jev-gc before the next LLM call. Not every LangGraph app is already
instrumented with OpenTelemetry, so this uses the **manual observe** path
(`JevGC.observe`) rather than requiring `attach_to_tracer_provider` --
if the host *is* emitting OTel spans for tool calls already, prefer
attaching jev-gc directly to that `TracerProvider` instead of this node,
since spans carry richer metadata (tool name, error type) than a bare
LangGraph message does.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from jevgc.gc import JevGC
from jevgc.models import SpanRecord, SpanStatus, now_unix_ns

NodeFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def jev_gc_node(
    gc: JevGC,
    *,
    messages_key: str = "messages",
    task_key: str = "task",
    context_key: str = "jevgc_context",
    budget_tokens: int = 8000,
) -> NodeFn:
    """Returns an async LangGraph node: `state -> state`.

    Reads `state[messages_key]`, feeds each message into jev-gc via the
    manual `observe` escape hatch, then writes the trimmed, budget-fit
    context string to `state[context_key]` for the next LLM call to read.
    """

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get(messages_key, [])
        task = state.get(task_key, "")

        for index, message in enumerate(messages):
            span = _message_to_span_record(message, index)
            await gc.observe(span)

        context = gc.build_context(task=task, budget_tokens=budget_tokens)
        return {**state, context_key: context}

    return node


def _message_to_span_record(message: Any, index: int) -> SpanRecord:
    """Best-effort conversion of a LangGraph/LangChain message object (or a
    plain dict/str) into a `SpanRecord`. Deliberately permissive via
    `getattr`/`.get` fallbacks since LangGraph state shapes vary by app."""
    if isinstance(message, dict):
        content = message.get("content", "")
        role = message.get("type") or message.get("role") or "message"
    else:
        content = getattr(message, "content", str(message))
        role = getattr(message, "type", None) or message.__class__.__name__

    content_str = str(content)
    timestamp = now_unix_ns()

    return SpanRecord(
        span_id=f"langgraph-msg-{index}",
        trace_id="langgraph-session",
        name=f"{role}_message",
        status=SpanStatus.OK,
        start_time_unix_ns=timestamp,
        end_time_unix_ns=timestamp,
        output_preview=content_str[:2000],
        output_token_count=max(1, len(content_str) // 4),
        turn_index=index,
    )
