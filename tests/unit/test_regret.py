from __future__ import annotations

from jevgc.archive import Archive, EvictionEvent
from jevgc.models import Tier
from jevgc.regret import find_regret


def _event(
    span_id: str,
    to_tier: Tier,
    *,
    from_tier: Tier | None = Tier.HOT,
    turn_index: int = 1,
    reason: str = "below_relevance_threshold treatment=drop",
    timestamp_unix_ns: int = 1_000,
) -> EvictionEvent:
    return EvictionEvent(
        span_id=span_id,
        from_tier=from_tier,
        to_tier=to_tier,
        turn_index=turn_index,
        reason=reason,
        timestamp_unix_ns=timestamp_unix_ns,
    )


def _archive_with(span_id: str, text: str, *, turn_index: int = 1) -> Archive:
    archive = Archive()
    archive.record(span_id, text, Tier.COLD, turn_index)
    return archive


def test_reports_regret_when_evicted_terms_appear_only_in_baseline():
    archive = _archive_with(
        "s1",
        "warehouse shipment WH-8812 was rerouted through the Rotterdam depot",
    )
    log = [_event("s1", Tier.COLD)]

    findings = find_regret(
        log,
        archive,
        baseline_output="The delay came from WH-8812 rerouting through the Rotterdam depot.",
        evicted_output="The delay came from an unspecified logistics issue.",
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.span_id == "s1"
    assert finding.evicted_to == Tier.COLD
    assert finding.eviction_reason == "below_relevance_threshold treatment=drop"
    assert finding.turn_index == 1
    assert len(finding.matched_keywords) >= 2
    assert 0.0 < finding.regret_score <= 1.0


def test_no_finding_when_terms_appear_in_both_outputs():
    shared = "warehouse shipment WH-8812 rerouted through the Rotterdam depot"
    archive = _archive_with("s1", shared)
    log = [_event("s1", Tier.WARM)]

    findings = find_regret(
        log,
        archive,
        baseline_output=f"Answer: {shared}.",
        evicted_output=f"Answer: {shared}.",
    )

    assert findings == []


def test_no_finding_when_span_was_promoted_back_to_hot():
    """Rehydration is the success case: the system recovered from its own
    eviction, so it must not be counted against it."""
    archive = _archive_with(
        "s1",
        "warehouse shipment WH-8812 rerouted through the Rotterdam depot",
    )
    log = [
        _event("s1", Tier.COLD, turn_index=1),
        _event("s1", Tier.HOT, from_tier=Tier.COLD, turn_index=3, reason="referenced_again"),
    ]

    findings = find_regret(
        log,
        archive,
        baseline_output="WH-8812 was rerouted through the Rotterdam depot.",
        evicted_output="No detail available.",
    )

    assert findings == []


def test_last_eviction_wins_when_span_is_evicted_again_after_promotion():
    archive = _archive_with(
        "s1",
        "warehouse shipment WH-8812 rerouted through the Rotterdam depot",
    )
    log = [
        _event("s1", Tier.WARM, turn_index=1),
        _event("s1", Tier.HOT, from_tier=Tier.WARM, turn_index=2, reason="referenced_again"),
        _event("s1", Tier.COLD, from_tier=Tier.HOT, turn_index=5, reason="budget_pressure"),
    ]

    findings = find_regret(
        log,
        archive,
        baseline_output="WH-8812 was rerouted through the Rotterdam depot.",
        evicted_output="No detail available.",
    )

    assert len(findings) == 1
    assert findings[0].evicted_to == Tier.COLD
    assert findings[0].eviction_reason == "budget_pressure"
    assert findings[0].turn_index == 5


def test_no_finding_below_min_keyword_hits():
    archive = _archive_with("s1", "Rotterdam depot inventory reconciliation summary")
    log = [_event("s1", Tier.COLD)]

    kwargs = {
        "baseline_output": "Only Rotterdam is mentioned here.",
        "evicted_output": "Nothing relevant.",
    }

    assert find_regret(log, archive, min_keyword_hits=2, **kwargs) == []
    assert len(find_regret(log, archive, min_keyword_hits=1, **kwargs)) == 1


def test_span_missing_from_archive_is_skipped_not_raised():
    archive = Archive()  # nothing recorded
    log = [_event("s1", Tier.COLD)]

    findings = find_regret(
        log,
        archive,
        baseline_output="WH-8812 rerouted through the Rotterdam depot.",
        evicted_output="No detail available.",
    )

    assert findings == []


def test_keywords_match_on_whole_tokens_only():
    """A keyword embedded inside a longer word is not a match -- substring
    matching (keyword 'ok' hitting inside 'broken') would manufacture regret
    out of unrelated prose."""
    archive = _archive_with("s1", "Rotterdam depot manifest customs")
    entry = archive.get("s1")
    assert entry is not None and entry.keywords, "precondition: archive extracted keywords"

    # Every keyword appears in the baseline, but only ever glued inside a
    # longer token -- so a whole-token matcher must find nothing.
    baseline = " ".join(f"xx{keyword}yy" for keyword in entry.keywords)

    findings = find_regret(
        [_event("s1", Tier.COLD)],
        archive,
        baseline_output=baseline,
        evicted_output="nothing",
        min_keyword_hits=1,
    )

    assert findings == []


def test_results_are_sorted_by_score_then_span_id():
    archive = Archive()
    # Two spans with identical keyword sets -> identical regret scores, so the
    # span_id tiebreak is what makes the order reproducible.
    archive.record("s2", "Rotterdam depot", Tier.COLD, 1)
    archive.record("s1", "Rotterdam depot", Tier.COLD, 1)
    # A third span where only half its keywords hit -> strictly lower score.
    archive.record("s0", "Rotterdam depot manifest customs", Tier.COLD, 1)

    log = [
        _event("s2", Tier.COLD),
        _event("s1", Tier.COLD),
        _event("s0", Tier.COLD),
    ]

    findings = find_regret(
        log,
        archive,
        baseline_output="Rotterdam depot was the cause.",
        evicted_output="Unknown cause.",
    )

    assert [f.span_id for f in findings] == ["s1", "s2", "s0"]
    assert findings[0].regret_score == findings[1].regret_score
    assert findings[2].regret_score < findings[1].regret_score


def test_promotion_only_log_yields_no_findings():
    archive = _archive_with("s1", "Rotterdam depot manifest")
    log = [_event("s1", Tier.HOT, from_tier=None, reason="first_assignment")]

    assert find_regret(
        log,
        archive,
        baseline_output="Rotterdam depot manifest details.",
        evicted_output="",
    ) == []
