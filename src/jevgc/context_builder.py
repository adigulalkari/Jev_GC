"""Renders `SpanRecord` + `GCDecision` pairs into `ContextItem`s, and
assembles the final prompt-ready context string (SPEC.md §3, §4.5/§4.8).

Token counting uses a cheap heuristic (~4 chars/token) rather than a real
tokenizer -- jev-gc has no tokenizer dependency, and the exact count only
needs to be consistent enough for the budget allocator's greedy knapsack to
behave sensibly, not perfectly accurate for any specific model.
"""

from __future__ import annotations

from jevgc.models import ContextItem, ErrorTreatment, GCDecision, SpanRecord, Treatment

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def full_content_text(span: SpanRecord) -> str:
    """What this span would render as if jev-gc never compressed or evicted
    anything -- i.e. `Treatment.INCLUDE_FULL` / `ErrorTreatment.KEEP_FULL_TRACE`
    rendering, used both by those branches below and as the "no GC at all"
    baseline for the token-savings stat in telemetry.py. Must match the
    *shape* (header + full body) of whatever treatment actually renders, or
    the savings comparison isn't apples to apples -- a short pointer string
    can otherwise look like it costs *more* than a body-less baseline.
    """
    header = f"[{span.name}]"
    if span.is_error:
        return _render_error(span, header, ErrorTreatment.KEEP_FULL_TRACE)
    body = span.output_preview or span.input_preview or "(no content)"
    return f"{header}\n{body}"


def full_content_tokens(span: SpanRecord) -> int:
    return estimate_tokens(full_content_text(span))


def render_span_content(span: SpanRecord, decision: GCDecision) -> str:
    """Renders a span's content according to its decided treatment."""
    header = f"[{span.name}]"

    if isinstance(decision.treatment, ErrorTreatment):
        return _render_error(span, header, decision.treatment)

    if decision.treatment == Treatment.INCLUDE_FULL:
        return full_content_text(span)

    if decision.treatment == Treatment.INCLUDE_SUMMARY_ONLY:
        body = _summarize(span.output_preview or span.input_preview or "")
        return f"{header} (summarized)\n{body}"

    if decision.treatment == Treatment.KEEP_POINTER_ONLY:
        return f"{header} -> evicted from prompt; retrievable via span_id={span.span_id}"

    # Treatment.DROP -- policy still produced a ContextItem (e.g. mid-transition
    # to COLD tier); render as a pointer rather than emitting nothing, since
    # jev-gc never hard-deletes (SPEC.md §1 non-goals).
    return f"{header} -> archived (span_id={span.span_id})"


def _render_error(span: SpanRecord, header: str, treatment: ErrorTreatment) -> str:
    error_type = span.error_type or "UnknownError"

    if treatment == ErrorTreatment.KEEP_FULL_TRACE:
        message = span.error_message or "(no message)"
        return f"{header} ERROR {error_type}: {message}"

    if treatment == ErrorTreatment.KEEP_ERROR_SUMMARY_ONLY:
        message = _summarize(span.error_message or "")
        return f"{header} ERROR {error_type} (summarized): {message}"

    # KEEP_ERROR_TYPE_ONLY
    return f"{header} ERROR {error_type} (details dropped)"


def _summarize(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_context_item(span: SpanRecord, decision: GCDecision) -> ContextItem:
    rendered_text = render_span_content(span, decision)
    return ContextItem(
        span_id=span.span_id,
        tier=decision.tier,
        rendered_text=rendered_text,
        token_count=estimate_tokens(rendered_text),
        decision=decision,
    )


def assemble_context(items: list[ContextItem]) -> str:
    """Joins selected items into the final prompt-ready text, HOT items in
    their given order (callers pass items already ordered/filtered by
    `policy.BudgetAllocator.allocate`)."""
    return "\n\n".join(item.rendered_text for item in items)
