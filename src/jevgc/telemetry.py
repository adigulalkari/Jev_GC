"""jev-gc's own self-observability (SPEC.md §3.2, §4.8 `GCStats`).

Tracks counters the host can inspect via `JevGC.stats()` and, when
`telemetry.emit_self_metrics` is enabled, mirrors them as OpenTelemetry
metrics instruments so they show up next to the host's own dashboards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from jevgc.models import GCDecision, Tier

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Meter


class GCStats(BaseModel):
    spans_processed: int = 0
    spans_kept_hot: int = 0
    spans_kept_warm: int = 0
    spans_archived_cold: int = 0
    jev_calls: int = 0
    jev_call_errors: int = 0
    tokens_rendered: int = 0


class JevGCTelemetry:
    """Plain in-process counters, optionally mirrored to OTel metrics.

    Mirroring is best-effort: if `opentelemetry.metrics` has no configured
    `MeterProvider`, instruments are created against the OTel no-op default
    and simply do nothing -- this module never fails because telemetry
    couldn't be wired up.
    """

    def __init__(self, emit_self_metrics: bool = True) -> None:
        self._stats = GCStats()
        self._emit_self_metrics = emit_self_metrics
        self._meter: Meter | None = None
        self._span_counter: Counter | None = None
        self._token_counter: Counter | None = None

        if emit_self_metrics:
            self._init_otel_instruments()

    def _init_otel_instruments(self) -> None:
        from opentelemetry import metrics

        self._meter = metrics.get_meter("jevgc")
        self._span_counter = self._meter.create_counter(
            "jevgc.spans_processed", description="Spans processed by jev-gc's GC pipeline"
        )
        self._token_counter = self._meter.create_counter(
            "jevgc.tokens_rendered", description="Tokens rendered into assembled context"
        )

    def record_decision(self, decision: GCDecision, token_count: int) -> None:
        self._stats.spans_processed += 1
        self._stats.tokens_rendered += token_count

        if decision.tier == Tier.HOT:
            self._stats.spans_kept_hot += 1
        elif decision.tier == Tier.WARM:
            self._stats.spans_kept_warm += 1
        else:
            self._stats.spans_archived_cold += 1

        if decision.used_jev:
            self._stats.jev_calls += 1

        if self._emit_self_metrics and self._span_counter is not None and self._token_counter is not None:
            self._span_counter.add(1, {"tier": decision.tier.value})
            self._token_counter.add(token_count, {"tier": decision.tier.value})

    def record_jev_error(self) -> None:
        self._stats.jev_call_errors += 1

    def snapshot(self) -> GCStats:
        return self._stats.model_copy()
