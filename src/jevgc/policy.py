"""DecisionPolicy + BudgetAllocator (SPEC.md §4.5).

Both are pure, synchronous, dependency-free functions of their inputs --
fully unit-testable with synthetic data, no Jev/store access. This module
is the only place a Jev answer is allowed to become a `GCDecision`; Jev's
raw answer never bypasses the confidence threshold or the fail-open rule
(SPEC.md §2 principle 2).
"""

from __future__ import annotations

from jevgc.config import PolicySettings
from jevgc.models import (
    ContextItem,
    ErrorTreatment,
    GCDecision,
    ItemRole,
    SpanRecord,
    Tier,
    Treatment,
)
from jevgc.scorer import JevSpanResult


class DecisionPolicy:
    def __init__(self, config: PolicySettings) -> None:
        self._config = config

    def decide(
        self,
        span: SpanRecord,
        jev_result: JevSpanResult | None,
        current_tier: Tier = Tier.HOT,
    ) -> GCDecision:
        if jev_result is None or not jev_result.used_jev:
            treatment: Treatment | ErrorTreatment = (
                ErrorTreatment.KEEP_FULL_TRACE if span.is_error else Treatment.INCLUDE_FULL
            )
            return GCDecision(
                span_id=span.span_id,
                tier=current_tier,
                treatment=treatment,
                reason="jev_unavailable_fail_open",
                used_jev=False,
            )

        if span.is_error:
            return self._decide_error(span, jev_result)

        return self._decide_success(span, jev_result)

    def _decide_error(self, span: SpanRecord, jev_result: JevSpanResult) -> GCDecision:
        if jev_result.choice is None:
            treatment = self._config.error_default_treatment
            tier = Tier.HOT if treatment == ErrorTreatment.KEEP_FULL_TRACE else Tier.WARM
            return GCDecision(
                span_id=span.span_id,
                tier=tier,
                treatment=treatment,
                reason="error_missing_choice_fail_open",
                used_jev=True,
            )

        treatment = ErrorTreatment(jev_result.choice.option)
        tier = Tier.HOT if treatment == ErrorTreatment.KEEP_FULL_TRACE else Tier.WARM
        return GCDecision(
            span_id=span.span_id,
            tier=tier,
            treatment=treatment,
            confidence=jev_result.choice.probability,
            role=_role_of(jev_result),
            reason=f"error_treatment={treatment.value}",
            used_jev=True,
        )

    def _decide_success(self, span: SpanRecord, jev_result: JevSpanResult) -> GCDecision:
        role = _role_of(jev_result)

        if jev_result.score is None:
            return GCDecision(
                span_id=span.span_id,
                tier=Tier.HOT,
                treatment=Treatment.INCLUDE_FULL,
                role=role,
                reason="missing_score_fail_open",
                used_jev=True,
            )

        if jev_result.score.confidence < self._config.min_confidence_to_drop:
            return GCDecision(
                span_id=span.span_id,
                tier=Tier.HOT,
                treatment=Treatment.INCLUDE_FULL,
                relevance_score=jev_result.score.score,
                confidence=jev_result.score.confidence,
                role=role,
                reason="low_confidence_fail_open",
                used_jev=True,
            )

        treatment = Treatment(jev_result.choice.option) if jev_result.choice else Treatment.INCLUDE_FULL

        if jev_result.score.score < self._config.relevance_keep_threshold:
            tier = Tier.COLD if treatment == Treatment.DROP else Tier.WARM
            reason = f"below_relevance_threshold treatment={treatment.value}"
        else:
            tier = Tier.HOT
            reason = f"above_relevance_threshold treatment={treatment.value}"

        return GCDecision(
            span_id=span.span_id,
            tier=tier,
            treatment=treatment,
            relevance_score=jev_result.score.score,
            confidence=jev_result.score.confidence,
            role=role,
            reason=reason,
            used_jev=True,
        )


def _role_of(jev_result: JevSpanResult) -> ItemRole | None:
    if jev_result.role is None:
        return None
    try:
        return ItemRole(jev_result.role.option)
    except ValueError:
        return None


class BudgetAllocator:
    """Greedy knapsack by `relevance_score` descending. Errored-span items
    are always included first regardless of score or budget (SPEC.md §2
    principle 5); remaining budget fills by descending relevance until
    exhausted."""

    @staticmethod
    def allocate(items: list[ContextItem], budget_tokens: int) -> list[ContextItem]:
        error_items = [item for item in items if isinstance(item.decision.treatment, ErrorTreatment)]
        other_items = [item for item in items if not isinstance(item.decision.treatment, ErrorTreatment)]
        other_items.sort(
            key=lambda item: item.decision.relevance_score
            if item.decision.relevance_score is not None
            else 0.0,
            reverse=True,
        )

        selected: list[ContextItem] = []
        remaining = budget_tokens

        for item in error_items:
            selected.append(item)
            remaining -= item.token_count

        for item in other_items:
            if remaining <= 0:
                break
            if item.token_count <= remaining:
                selected.append(item)
                remaining -= item.token_count

        return selected
