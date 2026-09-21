"""JevBatchScorer: batches ambiguous spans and asks Jev about them
(SPEC.md §4.4).

Two independent question tracks:
  Track A (successful spans): relevance (Score) + treatment (Choice) +
    optional item_role (Choice, config-gated).
  Track B (errored spans): error_treatment (Choice) only -- SPEC.md §2
    principle 5, errored spans are never scored for relevance, only for
    how much of their payload to keep.

Batching accumulates submitted spans until `batch_max_size` is reached or
`batch_max_wait_ms` elapses since the first item in the current batch,
whichever comes first, using an `asyncio.Task` timer (never a blocking
sleep) so it never stalls the caller's event loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from jevgc.exceptions import JevAPIError, JevGCError, JevTimeoutError
from jevgc.jev_client.client import JevBatchRequest, JevClient
from jevgc.models import (
    ErrorTreatment,
    ItemRole,
    JevChoiceAnswer,
    JevScoreAnswer,
    SpanRecord,
    Treatment,
)

logger = logging.getLogger(__name__)

_TREATMENT_DESCRIPTIONS: dict[str, str] = {
    Treatment.INCLUDE_FULL.value: "Keep the full content verbatim in the next prompt",
    Treatment.INCLUDE_SUMMARY_ONLY.value: "Keep only a short summary of this content",
    Treatment.KEEP_POINTER_ONLY.value: "Keep only a pointer; drop the content from the prompt",
    Treatment.DROP.value: "Evict entirely from the next prompt (still recoverable, never destroyed)",
}

_ERROR_TREATMENT_DESCRIPTIONS: dict[str, str] = {
    ErrorTreatment.KEEP_FULL_TRACE.value: "Keep the full error trace verbatim",
    ErrorTreatment.KEEP_ERROR_SUMMARY_ONLY.value: "Keep only a short summary of the error",
    ErrorTreatment.KEEP_ERROR_TYPE_ONLY.value: "Keep only the error type/category, drop details",
}

_ITEM_ROLE_DESCRIPTIONS: dict[str, str] = {
    ItemRole.CRITICAL_FACT.value: "A fact the current task's reasoning depends on",
    ItemRole.SUPPORTING_EVIDENCE.value: "Corroborates or adds color to a critical fact",
    ItemRole.DEAD_END.value: "An investigation path that did not pan out",
    ItemRole.DUPLICATE_OF_EXISTING.value: "Repeats information already present elsewhere",
    ItemRole.UNCERTAIN.value: "Does not clearly fit any of the other roles",
}

_Q_RELEVANCE = "relevance"
_Q_TREATMENT = "treatment"
_Q_ERROR_TREATMENT = "error_treatment"
_Q_ROLE = "role"


class JevSpanResult(BaseModel):
    """One span's merged Jev answers, as handed to `policy.DecisionPolicy`.

    Not a frozen contract in `models.py` because it's an internal seam
    between `scorer.py` and `policy.py` only -- `GCDecision` (models.py) is
    the type that crosses further downstream.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    span_id: str
    score: JevScoreAnswer | None = None
    choice: JevChoiceAnswer | None = None  # treatment or error_treatment
    role: JevChoiceAnswer | None = None
    used_jev: bool = True


def _fail_open_result(span: SpanRecord, item_id: str) -> JevSpanResult:
    return JevSpanResult(item_id=item_id, span_id=span.span_id, used_jev=False)


def _render_state(span: SpanRecord, task_context: str) -> str:
    parts = [
        f"current_task: {task_context}",
        f"span_name: {span.name}",
        f"status: {span.status.value}",
    ]
    if span.gen_ai_tool_name:
        parts.append(f"tool: {span.gen_ai_tool_name}")
    if span.input_preview:
        parts.append(f"input: {span.input_preview}")
    if span.output_preview:
        parts.append(f"output: {span.output_preview}")
    if span.is_error:
        parts.append(f"error_type: {span.error_type or 'unknown'}")
        parts.append(f"error_message: {span.error_message or ''}")
    return "\n".join(parts)


def _build_requests(span: SpanRecord, item_id: str, task_context: str, ask_item_role: bool) -> list[JevBatchRequest]:
    state = _render_state(span, task_context)

    if span.is_error:
        return [
            JevBatchRequest(
                item_id=item_id,
                question_id=_Q_ERROR_TREATMENT,
                type="choice",
                instructions=(
                    "This span represents a failed/errored operation. How much of its "
                    "error trace should be retained in the next prompt's context?"
                ),
                state=state,
                options=_ERROR_TREATMENT_DESCRIPTIONS,
            )
        ]

    requests = [
        JevBatchRequest(
            item_id=item_id,
            question_id=_Q_RELEVANCE,
            type="score",
            instructions="How relevant is this span's content to the current task?",
            state=state,
        ),
        JevBatchRequest(
            item_id=item_id,
            question_id=_Q_TREATMENT,
            type="choice",
            instructions="Given its relevance, how should this span's content be retained in context?",
            state=state,
            options=_TREATMENT_DESCRIPTIONS,
        ),
    ]
    if ask_item_role:
        requests.append(
            JevBatchRequest(
                item_id=item_id,
                question_id=_Q_ROLE,
                type="choice",
                instructions="What structural role does this span play relative to the current task?",
                state=state,
                options=_ITEM_ROLE_DESCRIPTIONS,
            )
        )
    return requests


class JevBatchScorer:
    def __init__(
        self,
        client: JevClient,
        batch_max_size: int = 50,
        batch_max_wait_ms: int = 200,
        ask_item_role: bool = False,
        on_batch_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._client = client
        self._batch_max_size = batch_max_size
        self._batch_max_wait_s = batch_max_wait_ms / 1000
        self._ask_item_role = ask_item_role
        self._on_batch_error = on_batch_error

        self._lock = asyncio.Lock()
        self._pending: list[tuple[SpanRecord, str, str, asyncio.Future[JevSpanResult]]] = []
        self._flush_timer: asyncio.Task[None] | None = None

        self.batch_error_count = 0
        self.batches_sent = 0

    async def submit(self, span: SpanRecord, *, item_id: str, task_context: str = "") -> JevSpanResult:
        """Enqueue a span for scoring; resolves once its batch is flushed."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[JevSpanResult] = loop.create_future()

        async with self._lock:
            self._pending.append((span, item_id, task_context, future))
            if len(self._pending) >= self._batch_max_size:
                await self._flush_locked()
            elif self._flush_timer is None:
                self._flush_timer = asyncio.create_task(self._flush_after_timeout())

        return await future

    async def force_flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def _flush_after_timeout(self) -> None:
        try:
            await asyncio.sleep(self._batch_max_wait_s)
        except asyncio.CancelledError:
            return
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        """Must be called while holding `self._lock`. Drains `self._pending`
        and schedules its execution as a background task so the lock is
        released immediately -- new submissions start a fresh batch instead
        of waiting on this batch's network round trip."""
        if not self._pending:
            return

        batch = self._pending
        self._pending = []

        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

        asyncio.create_task(self._execute_batch(batch))

    async def _execute_batch(
        self, batch: list[tuple[SpanRecord, str, str, asyncio.Future[JevSpanResult]]]
    ) -> None:
        requests: list[JevBatchRequest] = []
        for span, item_id, task_context, _future in batch:
            requests.extend(_build_requests(span, item_id, task_context, self._ask_item_role))

        try:
            raw_results = await self._client.batch_ask(requests)
        except (JevTimeoutError, JevAPIError, JevGCError) as exc:
            logger.warning("Jev batch of %d item(s) failed, failing open: %s", len(batch), exc)
            self.batch_error_count += 1
            if self._on_batch_error is not None:
                self._on_batch_error(exc)
            for span, item_id, _task_context, future in batch:
                if not future.done():
                    future.set_result(_fail_open_result(span, item_id))
            return

        self.batches_sent += 1
        by_item: dict[str, JevSpanResult] = {}
        for result in raw_results:
            existing = by_item.get(result.item_id)
            score = existing.score if existing else None
            choice = existing.choice if existing else None
            role = existing.role if existing else None

            if result.question_id == _Q_RELEVANCE:
                score = result.score
            elif result.question_id in (_Q_TREATMENT, _Q_ERROR_TREATMENT):
                choice = result.choice
            elif result.question_id == _Q_ROLE:
                role = result.choice

            span_id = existing.span_id if existing else _span_id_for(batch, result.item_id)
            by_item[result.item_id] = JevSpanResult(
                item_id=result.item_id,
                span_id=span_id,
                score=score,
                choice=choice,
                role=role,
                used_jev=True,
            )

        for span, item_id, _task_context, future in batch:
            if future.done():
                continue
            future.set_result(by_item.get(item_id, _fail_open_result(span, item_id)))


def _span_id_for(
    batch: list[tuple[SpanRecord, str, str, asyncio.Future[JevSpanResult]]], item_id: str
) -> str:
    for span, batch_item_id, _task_context, _future in batch:
        if batch_item_id == item_id:
            return span.span_id
    return ""
