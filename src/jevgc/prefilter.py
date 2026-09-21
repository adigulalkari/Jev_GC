"""Deterministic pre-filter (SPEC.md §4.3, §2 principle 1).

Pure functions, no I/O, no Jev dependency. Anything expressible as a
threshold/age/status/pin check belongs here, not in a Jev question -- Jev is
reserved for spans this module cannot confidently classify on its own.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict

from jevgc.models import SpanRecord


class PrefilterResult(str, enum.Enum):
    KEEP = "keep"
    DROP = "drop"
    AMBIGUOUS = "ambiguous"


class PrefilterContext(BaseModel):
    """Running context state the pre-filter rules need, supplied by the
    caller (typically the store/pipeline driver) -- never fetched by this
    module itself, to keep it pure and synchronous."""

    model_config = ConfigDict(frozen=True)

    current_turn_index: int
    pinned_span_ids: frozenset[str] = frozenset()
    referenced_span_ids: frozenset[str] = frozenset()
    keep_last_n_turns: int = 3
    drop_zero_output_after_turns: int = 10


_TRIVIAL_OUTPUT_TOKEN_THRESHOLD = 1


def apply_prefilter(span: SpanRecord, ctx: PrefilterContext) -> PrefilterResult:
    """Rules, in order -- first match wins (SPEC.md §4.3):

    1. pinned                                          -> KEEP
    2. span.is_error                                    -> KEEP
    3. referenced by a later span/turn                  -> KEEP
    4. within `keep_last_n_turns` of the current turn    -> KEEP
    5. old + OK + trivial output                         -> DROP
    6. otherwise                                          -> AMBIGUOUS
    """
    if span.span_id in ctx.pinned_span_ids:
        return PrefilterResult.KEEP

    if span.is_error:
        return PrefilterResult.KEEP

    if span.span_id in ctx.referenced_span_ids:
        return PrefilterResult.KEEP

    turns_ago = ctx.current_turn_index - span.turn_index
    if turns_ago <= ctx.keep_last_n_turns:
        return PrefilterResult.KEEP

    has_trivial_output = (
        span.output_token_count is None
        or span.output_token_count < _TRIVIAL_OUTPUT_TOKEN_THRESHOLD
    )
    if turns_ago > ctx.drop_zero_output_after_turns and has_trivial_output:
        return PrefilterResult.DROP

    return PrefilterResult.AMBIGUOUS
