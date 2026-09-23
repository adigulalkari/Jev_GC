from __future__ import annotations

from jevgc.models import ErrorTreatment, GCDecision, Tier, Treatment
from jevgc.telemetry import JevGCTelemetry


def _decision(span_id: str, tier: Tier, treatment: Treatment | ErrorTreatment) -> GCDecision:
    return GCDecision(span_id=span_id, tier=tier, treatment=treatment, reason="test", used_jev=False)


def test_include_full_contributes_no_savings():
    telemetry = JevGCTelemetry(emit_self_metrics=False)
    decision = _decision("s1", Tier.HOT, Treatment.INCLUDE_FULL)

    telemetry.record_decision(decision, token_count=100, full_token_count=100)

    stats = telemetry.snapshot()
    assert stats.tokens_rendered == 100
    assert stats.tokens_full_content == 100
    assert stats.tokens_saved_estimate == 0


def test_dropped_span_contributes_full_savings():
    telemetry = JevGCTelemetry(emit_self_metrics=False)
    decision = _decision("s2", Tier.COLD, Treatment.DROP)

    # A dropped/pointer-only span still renders a short pointer string
    # (context_builder never emits nothing), so token_count is small but
    # nonzero -- savings is the full content minus that small remainder.
    telemetry.record_decision(decision, token_count=5, full_token_count=200)

    stats = telemetry.snapshot()
    assert stats.tokens_rendered == 5
    assert stats.tokens_full_content == 200
    assert stats.tokens_saved_estimate == 195


def test_savings_accumulate_across_multiple_decisions():
    telemetry = JevGCTelemetry(emit_self_metrics=False)
    telemetry.record_decision(
        _decision("s1", Tier.HOT, Treatment.INCLUDE_FULL), token_count=50, full_token_count=50
    )
    telemetry.record_decision(
        _decision("s2", Tier.WARM, Treatment.KEEP_POINTER_ONLY), token_count=10, full_token_count=150
    )
    telemetry.record_decision(
        _decision("s3", Tier.HOT, ErrorTreatment.KEEP_FULL_TRACE), token_count=30, full_token_count=30
    )

    stats = telemetry.snapshot()
    assert stats.spans_processed == 3
    assert stats.tokens_rendered == 50 + 10 + 30
    assert stats.tokens_full_content == 50 + 150 + 30
    assert stats.tokens_saved_estimate == 140  # only s2 contributed savings


def test_rendering_larger_than_source_counts_against_savings():
    """A pointer can cost more than the short content it replaces. That is a
    real loss, and clamping it to zero would let the headline stat sum its
    wins while dropping its losses."""
    telemetry = JevGCTelemetry(emit_self_metrics=False)
    decision = _decision("s1", Tier.HOT, Treatment.INCLUDE_SUMMARY_ONLY)

    telemetry.record_decision(decision, token_count=20, full_token_count=10)

    stats = telemetry.snapshot()
    assert stats.tokens_saved_estimate == -10


def test_saved_estimate_always_equals_full_content_minus_rendered():
    """The invariant the field's docstring promises, checked against a mix
    where one span wins and another loses."""
    telemetry = JevGCTelemetry(emit_self_metrics=False)
    telemetry.record_decision(
        _decision("win", Tier.WARM, Treatment.KEEP_POINTER_ONLY),
        token_count=10,
        full_token_count=150,
    )
    telemetry.record_decision(
        _decision("loss", Tier.HOT, Treatment.INCLUDE_SUMMARY_ONLY),
        token_count=25,
        full_token_count=5,
    )

    stats = telemetry.snapshot()
    assert stats.tokens_saved_estimate == stats.tokens_full_content - stats.tokens_rendered
    assert stats.tokens_saved_estimate == 120


def test_snapshot_returns_independent_copy():
    telemetry = JevGCTelemetry(emit_self_metrics=False)
    telemetry.record_decision(
        _decision("s1", Tier.HOT, Treatment.INCLUDE_FULL), token_count=10, full_token_count=10
    )

    first = telemetry.snapshot()
    telemetry.record_decision(
        _decision("s2", Tier.HOT, Treatment.INCLUDE_FULL), token_count=10, full_token_count=10
    )

    assert first.spans_processed == 1  # unaffected by the later call
    assert telemetry.snapshot().spans_processed == 2
