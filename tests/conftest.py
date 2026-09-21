from __future__ import annotations

from typing import Any

import pytest

from jevgc.models import SpanRecord, SpanStatus


@pytest.fixture
def make_span():
    """Factory fixture: `make_span(turn_index=0, status=SpanStatus.OK, ...)`."""

    def _make(
        span_id: str = "span-1",
        turn_index: int = 0,
        status: SpanStatus = SpanStatus.OK,
        output_token_count: int | None = 50,
        output_preview: str | None = "some tool output",
        input_preview: str | None = "some tool input",
        error_type: str | None = None,
        error_message: str | None = None,
        name: str = "test_span",
        **overrides: Any,
    ) -> SpanRecord:
        return SpanRecord(
            span_id=span_id,
            trace_id="trace-1",
            name=name,
            status=status,
            start_time_unix_ns=0,
            end_time_unix_ns=1_000_000,
            output_token_count=output_token_count,
            output_preview=output_preview,
            input_preview=input_preview,
            error_type=error_type,
            error_message=error_message,
            turn_index=turn_index,
            **overrides,
        )

    return _make
