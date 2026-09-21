from __future__ import annotations

from jevgc.context_builder import assemble_context, build_context_item, estimate_tokens
from jevgc.models import ErrorTreatment, GCDecision, Tier, Treatment


def test_estimate_tokens_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_render_include_full(make_span):
    span = make_span(output_preview="the full answer")
    decision = GCDecision(
        span_id=span.span_id, tier=Tier.HOT, treatment=Treatment.INCLUDE_FULL, reason="t", used_jev=False
    )
    item = build_context_item(span, decision)
    assert "the full answer" in item.rendered_text
    assert item.tier == Tier.HOT


def test_render_pointer_only_mentions_span_id(make_span):
    span = make_span()
    decision = GCDecision(
        span_id=span.span_id, tier=Tier.WARM, treatment=Treatment.KEEP_POINTER_ONLY, reason="t", used_jev=True
    )
    item = build_context_item(span, decision)
    assert span.span_id in item.rendered_text
    assert "evicted" in item.rendered_text


def test_render_error_keeps_full_trace(make_span):
    from jevgc.models import SpanStatus

    span = make_span(status=SpanStatus.ERROR, error_type="ValueError", error_message="bad input: x=5")
    decision = GCDecision(
        span_id=span.span_id, tier=Tier.HOT, treatment=ErrorTreatment.KEEP_FULL_TRACE, reason="t", used_jev=True
    )
    item = build_context_item(span, decision)
    assert "ValueError" in item.rendered_text
    assert "bad input: x=5" in item.rendered_text


def test_render_error_type_only_drops_message(make_span):
    from jevgc.models import SpanStatus

    span = make_span(status=SpanStatus.ERROR, error_type="ValueError", error_message="bad input: x=5")
    decision = GCDecision(
        span_id=span.span_id,
        tier=Tier.WARM,
        treatment=ErrorTreatment.KEEP_ERROR_TYPE_ONLY,
        reason="t",
        used_jev=True,
    )
    item = build_context_item(span, decision)
    assert "ValueError" in item.rendered_text
    assert "bad input: x=5" not in item.rendered_text


def test_assemble_context_joins_items(make_span):
    span1, span2 = make_span(span_id="a"), make_span(span_id="b")
    decision = GCDecision(span_id="a", tier=Tier.HOT, treatment=Treatment.INCLUDE_FULL, reason="t", used_jev=False)
    item1 = build_context_item(span1, decision)
    item2 = build_context_item(span2, decision.model_copy(update={"span_id": "b"}))

    text = assemble_context([item1, item2])
    assert item1.rendered_text in text
    assert item2.rendered_text in text
