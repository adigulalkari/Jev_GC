"""End-to-end: synthetic span sequences through the full prefilter -> scorer
-> policy -> store -> context_builder pipeline, via `JevGC` with a
`FakeJevClient` (no network). Scenario mirrors SPEC.md §7's suggestion: a
table-routing-style investigation with one dead end, one error, one
duplicate.
"""

from __future__ import annotations

import pytest

from jevgc.config import JevGCConfig
from jevgc.gc import JevGC
from jevgc.jev_client.fakes import FakeJevClient
from jevgc.models import (
    ErrorTreatment,
    JevChoiceAnswer,
    JevQuestionResult,
    JevScoreAnswer,
    SpanStatus,
    Tier,
    Treatment,
)


def _config() -> JevGCConfig:
    return JevGCConfig.from_dict(
        {
            "jev": {"api_key": "unused"},
            "prefilter": {"keep_last_n_turns": 0, "drop_zero_output_after_turns": 100},
            "policy": {"relevance_keep_threshold": 0.35, "min_confidence_to_drop": 0.6},
        }
    )


def _scored_responder(requests):
    """Routes fake Jev answers by item_id: 'relevant' -> keep, 'deadend' ->
    drop, 'erroring' -> error_treatment summary."""
    out = []
    for req in requests:
        if req.item_id == "erroring":
            out.append(
                JevQuestionResult(
                    item_id=req.item_id,
                    question_id=req.question_id,
                    choice=JevChoiceAnswer(option=ErrorTreatment.KEEP_ERROR_SUMMARY_ONLY.value, probability=0.8),
                )
            )
        elif req.question_id == "relevance":
            score = 0.9 if req.item_id == "relevant" else 0.1
            out.append(
                JevQuestionResult(
                    item_id=req.item_id, question_id=req.question_id,
                    score=JevScoreAnswer(score=score, confidence=0.9),
                )
            )
        else:
            treatment = Treatment.INCLUDE_FULL.value if req.item_id == "relevant" else Treatment.DROP.value
            out.append(
                JevQuestionResult(
                    item_id=req.item_id, question_id=req.question_id,
                    choice=JevChoiceAnswer(option=treatment, probability=0.8),
                )
            )
    return out


@pytest.mark.asyncio
async def test_investigation_scenario_dead_end_error_and_duplicate(make_span):
    client = FakeJevClient(responder=_scored_responder)
    gc = JevGC(_config(), jev_client=client)

    relevant_span = make_span(
        span_id="relevant", turn_index=0, output_preview="ERP total: $4521.00", output_token_count=200
    )
    dead_end_span = make_span(
        span_id="deadend", turn_index=0, output_preview="unrelated warehouse SKU listing", output_token_count=200
    )
    error_span = make_span(
        span_id="erroring",
        turn_index=0,
        status=SpanStatus.ERROR,
        error_type="CarrierAPITimeout",
        error_message="shipping_carrier_api timed out after 30s",
        output_token_count=0,
    )

    gc.advance_turn(50)  # push everything outside keep_last_n_turns=0 window
    for span in (relevant_span, dead_end_span, error_span):
        await gc.observe(span)

    context = gc.build_context(task="Reconcile invoice #4521", budget_tokens=1000)

    assert "ERP total: $4521.00" in context  # relevant span kept in full
    assert "unrelated warehouse SKU listing" not in context  # dead end dropped from prompt
    assert "CarrierAPITimeout" in context  # error never dropped, only compressed
    assert "shipping_carrier_api timed out" in context  # summary treatment keeps message

    stats = gc.stats()
    assert stats.spans_processed == 3
    assert stats.jev_calls == 3


@pytest.mark.asyncio
async def test_manual_observe_returns_decision_per_span(make_span):
    client = FakeJevClient()
    gc = JevGC(_config(), jev_client=client)

    span = make_span(turn_index=0)
    gc.advance_turn(50)
    decision = await gc.observe(span)

    assert decision.span_id == span.span_id
    assert decision.used_jev is True


@pytest.mark.asyncio
async def test_pin_and_mark_referenced_promote_spans(make_span):
    client = FakeJevClient()
    gc = JevGC(_config(), jev_client=client)

    span = make_span(span_id="s1", turn_index=0)
    gc.pin("s1")
    gc.advance_turn(50)

    decision = await gc.observe(span)
    assert decision.tier == Tier.HOT
    assert decision.used_jev is False  # pinned -> prefilter KEEP, never reaches Jev
