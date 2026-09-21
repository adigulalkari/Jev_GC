from __future__ import annotations

import pytest

from jevgc.config import PolicySettings
from jevgc.models import (
    ContextItem,
    ErrorTreatment,
    GCDecision,
    JevChoiceAnswer,
    JevScoreAnswer,
    SpanStatus,
    Tier,
    Treatment,
)
from jevgc.policy import BudgetAllocator, DecisionPolicy
from jevgc.scorer import JevSpanResult


@pytest.fixture
def policy() -> DecisionPolicy:
    return DecisionPolicy(PolicySettings(relevance_keep_threshold=0.35, min_confidence_to_drop=0.6))


def test_fail_open_on_none_jev_result(policy, make_span):
    span = make_span()
    decision = policy.decide(span, None)
    assert decision.used_jev is False
    assert decision.tier == Tier.HOT
    assert decision.reason == "jev_unavailable_fail_open"


def test_fail_open_when_scorer_marks_not_used_jev(policy, make_span):
    span = make_span()
    result = JevSpanResult(item_id="i1", span_id=span.span_id, used_jev=False)
    decision = policy.decide(span, result)
    assert decision.used_jev is False
    assert decision.tier == Tier.HOT


def test_fail_open_on_low_confidence(policy, make_span):
    span = make_span()
    result = JevSpanResult(
        item_id="i1",
        span_id=span.span_id,
        score=JevScoreAnswer(score=0.1, confidence=0.2),
        choice=JevChoiceAnswer(option=Treatment.DROP.value, probability=0.9),
    )
    decision = policy.decide(span, result)
    assert decision.tier == Tier.HOT
    assert decision.reason == "low_confidence_fail_open"
    assert decision.used_jev is True


@pytest.mark.parametrize(
    "score,treatment,expected_tier",
    [
        (0.9, Treatment.INCLUDE_FULL, Tier.HOT),
        (0.9, Treatment.DROP, Tier.HOT),  # above threshold always -> HOT regardless of treatment
        (0.1, Treatment.DROP, Tier.COLD),
        (0.1, Treatment.KEEP_POINTER_ONLY, Tier.WARM),
        (0.1, Treatment.INCLUDE_SUMMARY_ONLY, Tier.WARM),
    ],
)
def test_tier_assignment_for_successful_spans(policy, make_span, score, treatment, expected_tier):
    span = make_span()
    result = JevSpanResult(
        item_id="i1",
        span_id=span.span_id,
        score=JevScoreAnswer(score=score, confidence=0.9),
        choice=JevChoiceAnswer(option=treatment.value, probability=0.8),
    )
    decision = policy.decide(span, result)
    assert decision.tier == expected_tier
    assert decision.treatment == treatment


def test_missing_score_fails_open(policy, make_span):
    span = make_span()
    result = JevSpanResult(item_id="i1", span_id=span.span_id, score=None, choice=None)
    decision = policy.decide(span, result)
    assert decision.tier == Tier.HOT
    assert decision.reason == "missing_score_fail_open"


@pytest.mark.parametrize(
    "treatment,expected_tier",
    [
        (ErrorTreatment.KEEP_FULL_TRACE, Tier.HOT),
        (ErrorTreatment.KEEP_ERROR_SUMMARY_ONLY, Tier.WARM),
        (ErrorTreatment.KEEP_ERROR_TYPE_ONLY, Tier.WARM),
    ],
)
def test_error_treatment_tier_assignment(policy, make_span, treatment, expected_tier):
    span = make_span(status=SpanStatus.ERROR, error_type="ValueError")
    result = JevSpanResult(
        item_id="i1",
        span_id=span.span_id,
        choice=JevChoiceAnswer(option=treatment.value, probability=0.7),
    )
    decision = policy.decide(span, result)
    assert decision.tier == expected_tier
    assert decision.treatment == treatment


def test_error_missing_choice_falls_back_to_config_default(policy, make_span):
    span = make_span(status=SpanStatus.ERROR, error_type="ValueError")
    result = JevSpanResult(item_id="i1", span_id=span.span_id, choice=None)
    decision = policy.decide(span, result)
    assert decision.treatment == ErrorTreatment.KEEP_ERROR_SUMMARY_ONLY
    assert decision.tier == Tier.WARM
    assert decision.reason == "error_missing_choice_fail_open"


def _item(span_id: str, tier: Tier, treatment, relevance: float | None, tokens: int) -> ContextItem:
    return ContextItem(
        span_id=span_id,
        tier=tier,
        rendered_text="x" * tokens,
        token_count=tokens,
        decision=GCDecision(
            span_id=span_id,
            tier=tier,
            treatment=treatment,
            relevance_score=relevance,
            reason="test",
            used_jev=True,
        ),
    )


def test_budget_allocator_always_includes_error_items_first():
    error_item = _item("e1", Tier.HOT, ErrorTreatment.KEEP_FULL_TRACE, None, tokens=100)
    low_relevance_item = _item("s1", Tier.WARM, Treatment.KEEP_POINTER_ONLY, 0.1, tokens=50)

    selected = BudgetAllocator.allocate([low_relevance_item, error_item], budget_tokens=10)

    assert error_item in selected
    # error item alone already exceeds budget, so the other item is excluded
    assert low_relevance_item not in selected


def test_budget_allocator_fills_by_descending_relevance():
    low = _item("s1", Tier.HOT, Treatment.INCLUDE_FULL, 0.2, tokens=50)
    high = _item("s2", Tier.HOT, Treatment.INCLUDE_FULL, 0.9, tokens=50)
    mid = _item("s3", Tier.HOT, Treatment.INCLUDE_FULL, 0.5, tokens=50)

    selected = BudgetAllocator.allocate([low, high, mid], budget_tokens=100)

    assert selected == [high, mid]


def test_budget_allocator_skips_items_that_dont_fit_but_keeps_smaller_later_ones():
    huge = _item("s1", Tier.HOT, Treatment.INCLUDE_FULL, 0.9, tokens=1000)
    small = _item("s2", Tier.HOT, Treatment.INCLUDE_FULL, 0.5, tokens=10)

    selected = BudgetAllocator.allocate([huge, small], budget_tokens=100)

    assert huge not in selected
    assert small in selected
