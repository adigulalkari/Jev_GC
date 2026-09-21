from __future__ import annotations

import pytest

from jevgc.models import SpanStatus
from jevgc.prefilter import PrefilterContext, PrefilterResult, apply_prefilter


def _ctx(**overrides) -> PrefilterContext:
    defaults: dict = {
        "current_turn_index": 10,
        "pinned_span_ids": frozenset(),
        "referenced_span_ids": frozenset(),
        "keep_last_n_turns": 3,
        "drop_zero_output_after_turns": 5,
    }
    defaults.update(overrides)
    return PrefilterContext(**defaults)


def test_pinned_span_is_kept_even_if_old_and_trivial(make_span):
    span = make_span(span_id="p1", turn_index=0, output_token_count=0)
    ctx = _ctx(pinned_span_ids=frozenset({"p1"}))
    assert apply_prefilter(span, ctx) == PrefilterResult.KEEP


def test_error_span_is_kept_even_if_old_and_trivial(make_span):
    span = make_span(span_id="e1", turn_index=0, status=SpanStatus.ERROR, output_token_count=0)
    ctx = _ctx()
    assert apply_prefilter(span, ctx) == PrefilterResult.KEEP


def test_referenced_span_is_kept(make_span):
    span = make_span(span_id="r1", turn_index=0, output_token_count=0)
    ctx = _ctx(referenced_span_ids=frozenset({"r1"}))
    assert apply_prefilter(span, ctx) == PrefilterResult.KEEP


def test_recent_span_within_keep_last_n_turns_is_kept(make_span):
    span = make_span(turn_index=8)  # current_turn_index=10, keep_last_n_turns=3
    ctx = _ctx()
    assert apply_prefilter(span, ctx) == PrefilterResult.KEEP


def test_old_ok_trivial_output_is_dropped(make_span):
    span = make_span(turn_index=1, output_token_count=0)  # 9 turns ago > drop_after=5
    ctx = _ctx()
    assert apply_prefilter(span, ctx) == PrefilterResult.DROP


def test_old_ok_trivial_output_none_token_count_is_dropped(make_span):
    span = make_span(turn_index=1, output_token_count=None)
    ctx = _ctx()
    assert apply_prefilter(span, ctx) == PrefilterResult.DROP


def test_old_ok_nontrivial_output_is_ambiguous(make_span):
    span = make_span(turn_index=1, output_token_count=500)
    ctx = _ctx()
    assert apply_prefilter(span, ctx) == PrefilterResult.AMBIGUOUS


def test_ordering_pin_beats_drop_conditions(make_span):
    span = make_span(span_id="p2", turn_index=1, output_token_count=0)
    ctx = _ctx(pinned_span_ids=frozenset({"p2"}))
    assert apply_prefilter(span, ctx) == PrefilterResult.KEEP


def test_ordering_error_beats_referenced_and_recency(make_span):
    # an error span far in the past, not pinned/referenced -- still KEEP via rule 2
    span = make_span(turn_index=0, status=SpanStatus.ERROR, output_token_count=0)
    ctx = _ctx(current_turn_index=100)
    assert apply_prefilter(span, ctx) == PrefilterResult.KEEP


def test_ordering_referenced_beats_age_drop(make_span):
    span = make_span(span_id="r2", turn_index=0, output_token_count=0)
    ctx = _ctx(current_turn_index=100, referenced_span_ids=frozenset({"r2"}))
    assert apply_prefilter(span, ctx) == PrefilterResult.KEEP


@pytest.mark.parametrize("turns_ago,expected", [(5, PrefilterResult.KEEP), (6, PrefilterResult.AMBIGUOUS)])
def test_boundary_at_keep_last_n_turns(make_span, turns_ago, expected):
    span = make_span(turn_index=10 - turns_ago, output_token_count=500)
    ctx = _ctx(current_turn_index=10, keep_last_n_turns=5)
    assert apply_prefilter(span, ctx) == expected
