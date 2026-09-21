"""Strands Agents integration (SPEC.md §4.10).

Strands Agents ships native OpenTelemetry instrumentation using the GenAI
semantic conventions out of the box, so this integration is intentionally
thin: pull the agent's configured `TracerProvider` and attach jev-gc's span
processor to it via `JevGC.attach_to_tracer_provider`.

Gotchas to keep in mind when tuning `otel/translate.py` against Strands
spans specifically:

- Strands names tool-call spans `"execute_tool <tool_name>"` rather than
  just the tool name -- `otel/translate.py` reads `gen_ai.tool.name` from
  attributes for `SpanRecord.gen_ai_tool_name`, not the span name, so this
  doesn't need special-casing, but don't rely on `SpanRecord.name` alone to
  identify a tool call.
- Strands emits a separate span per model "cycle" (including retries within
  a single logical turn), which can inflate `turn_index` bookkeeping if the
  host increments turns per span rather than per user-visible agent step --
  prefer driving `JevGC.advance_turn()` from the host's own step loop, not
  from span volume.
- Errors surfaced by Strands tool wrappers set both `error.type` and a
  `status.description`; jev-gc's translator prefers `status.description`
  for `SpanRecord.error_message`, which matches what Strands populates.
"""

from __future__ import annotations

from typing import Any

from jevgc.gc import JevGC


def attach_strands_agent(gc: JevGC, agent: Any) -> None:
    """Attaches `gc` to `agent`'s configured `TracerProvider`.

    Falls back to the process-global `TracerProvider` (via
    `opentelemetry.trace.get_tracer_provider()`) if the agent object doesn't
    expose one directly -- Strands' own tracer setup registers itself
    globally in common configurations.
    """
    tracer_provider = _resolve_tracer_provider(agent)
    gc.attach_to_tracer_provider(tracer_provider)


def _resolve_tracer_provider(agent: Any) -> Any:
    tracer = getattr(agent, "tracer", None)
    tracer_provider = getattr(tracer, "tracer_provider", None)
    if tracer_provider is not None:
        return tracer_provider

    from opentelemetry import trace

    return trace.get_tracer_provider()
