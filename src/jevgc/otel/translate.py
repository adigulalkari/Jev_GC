"""`ReadableSpan` -> `SpanRecord` translation (SPEC.md §4.7).

Extracts GenAI semantic-convention attributes where present and degrades
gracefully for spans that don't carry them (e.g. a raw DB call span) --
every field beyond the OTel-guaranteed ones is optional on `SpanRecord`.
"""

from __future__ import annotations

import json

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from jevgc.context_builder import estimate_tokens
from jevgc.models import SpanRecord, SpanStatus
from jevgc.otel.attributes import (
    ERROR_TYPE,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    JEVGC_INPUT_PREVIEW,
    JEVGC_OUTPUT_PREVIEW,
    JEVGC_TURN_INDEX,
)

_PREVIEW_MAX_CHARS = 2000

#: GenAI semantic-convention event names, in priority order, whose
#: `content`/`message` attribute holds this span's *input*. Strands (and
#: other GenAI-convention-instrumented hosts) emit these as span events
#: rather than attributes -- see docs/architecture.md.
_INPUT_EVENT_NAMES = ("gen_ai.tool.message", "gen_ai.user.message")
#: ...and this span's *output*.
_OUTPUT_EVENT_NAMES = ("gen_ai.choice",)

_STATUS_MAP = {
    StatusCode.OK: SpanStatus.OK,
    StatusCode.ERROR: SpanStatus.ERROR,
    StatusCode.UNSET: SpanStatus.UNSET,
}


def readable_span_to_record(span: ReadableSpan, *, default_turn_index: int = 0) -> SpanRecord:
    """Raises on truly malformed spans (missing context/timestamps); the
    caller (`JevGCSpanProcessor.on_end`) is responsible for catching and
    converting that into a logged, skipped span rather than a crash."""
    context = span.context
    if context is None:
        raise ValueError(f"Span {span.name!r} has no SpanContext; cannot translate")

    attributes = dict(span.attributes or {})
    status_code = span.status.status_code if span.status is not None else StatusCode.UNSET

    parent_span_id = None
    if span.parent is not None:
        parent_span_id = format(span.parent.span_id, "016x")

    # jev-gc's own `jevgc.*` attributes (a host can set these explicitly)
    # take priority; falling back to the GenAI semantic-convention *events*
    # that instrumentation like Strands' actually emits, since those never
    # land in `span.attributes` at all.
    input_preview = _as_str(attributes.get(JEVGC_INPUT_PREVIEW)) or _event_text(span, _INPUT_EVENT_NAMES)
    output_preview = _as_str(attributes.get(JEVGC_OUTPUT_PREVIEW)) or _event_text(span, _OUTPUT_EVENT_NAMES)

    output_token_count = _as_int(attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS))
    if output_token_count is None and output_preview:
        output_token_count = estimate_tokens(output_preview)

    return SpanRecord(
        span_id=format(context.span_id, "016x"),
        trace_id=format(context.trace_id, "032x"),
        parent_span_id=parent_span_id,
        name=span.name,
        status=_STATUS_MAP.get(status_code, SpanStatus.UNSET),
        start_time_unix_ns=span.start_time or 0,
        end_time_unix_ns=span.end_time or span.start_time or 0,
        attributes=attributes,
        gen_ai_operation_name=attributes.get("gen_ai.operation.name"),
        gen_ai_tool_name=attributes.get(GEN_AI_TOOL_NAME),
        error_type=attributes.get(ERROR_TYPE),
        error_message=_error_message(span),
        input_preview=input_preview,
        output_preview=output_preview,
        output_token_count=output_token_count,
        turn_index=_turn_index_or_default(attributes.get(JEVGC_TURN_INDEX), default_turn_index),
    )


def _event_text(span: ReadableSpan, event_names: tuple[str, ...]) -> str | None:
    events_by_name = {event.name: event for event in span.events or []}
    for name in event_names:
        event = events_by_name.get(name)
        if event is None or not event.attributes:
            continue
        raw = event.attributes.get("message") or event.attributes.get("content")
        if raw:
            return _flatten_message_json(str(raw))
    return None


def _flatten_message_json(raw: str) -> str:
    """GenAI-convention event payloads are JSON strings, usually a list of
    content blocks (`[{"text": "..."}]`, `[{"toolUse": {...}}]`, etc.).
    Pulls out every `"text"` field found anywhere in the structure; falls
    back to the raw (truncated) string if it isn't JSON or has no text."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[:_PREVIEW_MAX_CHARS]

    texts: list[str] = []
    _collect_text_fields(parsed, texts)
    if texts:
        return " ".join(texts)[:_PREVIEW_MAX_CHARS]
    return raw[:_PREVIEW_MAX_CHARS]


def _collect_text_fields(node: object, out: list[str]) -> None:
    if isinstance(node, dict):
        text = node.get("text")
        if isinstance(text, str):
            out.append(text)
        for value in node.values():
            _collect_text_fields(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_text_fields(item, out)


def _turn_index_or_default(value: object, default: int) -> int:
    parsed = _as_int(value)
    return default if parsed is None else parsed


def _as_str(value: object) -> str | None:
    return None if value is None else str(value)


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    return None


def _error_message(span: ReadableSpan) -> str | None:
    if span.status is not None and span.status.description:
        return span.status.description
    for event in span.events or []:
        if event.name == "exception":
            message = event.attributes.get("exception.message") if event.attributes else None
            if message:
                return str(message)
    return None
