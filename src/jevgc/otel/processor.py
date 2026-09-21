"""JevGCSpanProcessor (SPEC.md §4.7): the OTel `SpanProcessor` entry point.

`on_end` must never raise into the SDK's export pipeline -- a translation
failure is logged, counted, and the span is skipped. Handing the resulting
`SpanRecord` to the prefilter/scorer/policy/store pipeline is done via an
async callback scheduled on the running event loop, so a slow Jev call
never blocks span export.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from jevgc.exceptions import OTelIntegrationError
from jevgc.models import SpanRecord
from jevgc.otel.translate import readable_span_to_record

logger = logging.getLogger(__name__)

SpanHandler = Callable[[SpanRecord], Awaitable[None]]


class JevGCSpanProcessor(SpanProcessor):
    """Translates ended spans to `SpanRecord`s and dispatches them to
    `on_span_record` (typically `JevGC._observe`, wired up by
    `JevGC.attach_to_tracer_provider`).

    If no event loop is running when a span ends (e.g. spans emitted from
    sync code with no loop yet started), records are buffered in
    `pending_records` and can be drained later via `drain_pending`.
    """

    def __init__(
        self,
        on_span_record: SpanHandler | None = None,
        turn_index_fn: Callable[[ReadableSpan], int] | None = None,
    ) -> None:
        self._on_span_record = on_span_record
        self._turn_index_fn = turn_index_fn
        self.pending_records: list[SpanRecord] = []
        self.translation_error_count = 0
        self._in_flight: set[asyncio.Task[None]] = set()

    def on_start(self, span: ReadableSpan, parent_context: Context | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        default_turn_index = self._turn_index_fn(span) if self._turn_index_fn else 0
        try:
            record = readable_span_to_record(span, default_turn_index=default_turn_index)
        except Exception as exc:  # noqa: BLE001 - must never propagate into the SDK
            self.translation_error_count += 1
            integration_error = OTelIntegrationError(
                f"Failed to translate span {getattr(span, 'name', '?')!r}: {exc}"
            )
            logger.warning(str(integration_error))
            return

        self._dispatch(record)

    def _dispatch(self, record: SpanRecord) -> None:
        if self._on_span_record is None:
            self.pending_records.append(record)
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.pending_records.append(record)
            return

        task = loop.create_task(self._safe_handle(record))
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)

    async def _safe_handle(self, record: SpanRecord) -> None:
        assert self._on_span_record is not None
        try:
            await self._on_span_record(record)
        except Exception:  # noqa: BLE001 - never crash the caller's loop
            logger.exception("jev-gc span handler raised for span_id=%s", record.span_id)

    async def drain_pending(self) -> None:
        """Flush any records buffered while no event loop was running."""
        if self._on_span_record is None:
            return
        records, self.pending_records = self.pending_records, []
        for record in records:
            await self._safe_handle(record)

    async def wait_all(self) -> None:
        """Awaits every in-flight dispatch task, then drains anything that
        was buffered before a loop existed. Useful at the end of a run
        (e.g. an example script) to make sure every span has been observed
        before reading `JevGC.stats()` / `build_context`."""
        while self._in_flight:
            await asyncio.gather(*list(self._in_flight), return_exceptions=True)
        await self.drain_pending()

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True
