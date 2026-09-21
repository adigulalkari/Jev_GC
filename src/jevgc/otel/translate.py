"""`ReadableSpan` -> `SpanRecord` translation (SPEC.md §4.7).

Extracts GenAI semantic-convention attributes where present and degrades
gracefully for spans that don't carry them (e.g. a raw DB call span) --
every field beyond the OTel-guaranteed ones is optional on `SpanRecord`.
"""

from __future__ import annotations

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from jevgc.models import SpanRecord, SpanStatus
from jevgc.otel.attributes import (
    ERROR_TYPE,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    JEVGC_INPUT_PREVIEW,
    JEVGC_OUTPUT_PREVIEW,
    JEVGC_TURN_INDEX,
)

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
        input_preview=_as_str(attributes.get(JEVGC_INPUT_PREVIEW)),
        output_preview=_as_str(attributes.get(JEVGC_OUTPUT_PREVIEW)),
        output_token_count=_as_int(attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS)),
        turn_index=_turn_index_or_default(attributes.get(JEVGC_TURN_INDEX), default_turn_index),
    )


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
