"""`JevGC` -- the public facade (SPEC.md §4.8). Keep its surface small and
stable; everything else in the package is an implementation detail behind
it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from jevgc.config import JevGCConfig
from jevgc.context_builder import assemble_context, build_context_item
from jevgc.jev_client.client import HTTPJevClient, JevClient
from jevgc.models import GCDecision, SpanRecord, Tier, Treatment
from jevgc.policy import BudgetAllocator, DecisionPolicy
from jevgc.prefilter import PrefilterContext, PrefilterResult, apply_prefilter
from jevgc.scorer import JevBatchScorer
from jevgc.store import Backend, TieredContextStore
from jevgc.telemetry import GCStats, JevGCTelemetry

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from jevgc.otel.processor import JevGCSpanProcessor

logger = logging.getLogger(__name__)


class JevGC:
    """Real-time context garbage collector.

    Wire it to an OTel `TracerProvider` (`attach_to_tracer_provider`) for
    automatic span observation, or call `observe` directly for hosts that
    aren't already emitting OTel spans (SPEC.md §4.9's "manual observe"
    escape hatch). Call `build_context` whenever the host needs the next
    prompt's context string.
    """

    def __init__(
        self,
        config: JevGCConfig,
        jev_client: JevClient | None = None,
        backend: Backend | None = None,
    ) -> None:
        self._config = config
        self._owns_client = jev_client is None
        self._client: JevClient = jev_client or HTTPJevClient(
            api_key=config.jev.api_key,
            base_url=config.jev.base_url,
            timeout_seconds=config.jev.timeout_seconds,
            max_retries=config.jev.max_retries,
        )
        self._scorer = JevBatchScorer(
            self._client,
            batch_max_size=config.scorer.batch_max_size,
            batch_max_wait_ms=config.scorer.batch_max_wait_ms,
            on_batch_error=lambda _exc: self._telemetry.record_jev_error(),
        )
        self._policy = DecisionPolicy(config.policy)
        if backend is None:
            from jevgc.backends.memory import InMemoryBackend

            backend = InMemoryBackend()
        self._store = TieredContextStore(backend)
        self._telemetry = JevGCTelemetry(emit_self_metrics=config.telemetry.emit_self_metrics)

        self._processor: JevGCSpanProcessor | None = None
        self._turn_index = 0
        self._current_task = ""
        self._pinned_span_ids: set[str] = set()
        self._referenced_span_ids: set[str] = set()

    @classmethod
    def from_config(cls, path: str | Path) -> JevGC:
        return cls(JevGCConfig.from_config(path))

    def attach_to_tracer_provider(self, provider: TracerProvider) -> None:
        """Registers a `JevGCSpanProcessor` on `provider` so every span it
        exports flows through jev-gc's pipeline automatically."""
        from jevgc.otel.processor import JevGCSpanProcessor

        self._processor = JevGCSpanProcessor(
            on_span_record=self._handle_span_record,
            turn_index_fn=lambda _span: self._turn_index,
        )
        provider.add_span_processor(self._processor)

    async def _handle_span_record(self, record: SpanRecord) -> None:
        await self.observe(record)

    async def observe(self, span: SpanRecord) -> GCDecision:
        """Manual escape hatch (SPEC.md §4.9): run one `SpanRecord` through
        the prefilter -> scorer -> policy -> store pipeline and return its
        decision. Hosts not emitting OTel spans call this directly."""
        prefilter_ctx = PrefilterContext(
            current_turn_index=self._turn_index,
            pinned_span_ids=frozenset(self._pinned_span_ids),
            referenced_span_ids=frozenset(self._referenced_span_ids),
            keep_last_n_turns=self._config.prefilter.keep_last_n_turns,
            drop_zero_output_after_turns=self._config.prefilter.drop_zero_output_after_turns,
        )
        result = apply_prefilter(span, prefilter_ctx)

        # An error span always matches prefilter rule 2 (KEEP) or an earlier
        # rule (pin/reference) that also evaluates to KEEP -- either way its
        # *presence* is guaranteed. But per SPEC.md §2 principle 5, only its
        # *payload size* is negotiable via Jev, so error spans still route to
        # the scorer's error-treatment track (§4.4 Track B) rather than
        # short-circuiting like a plain non-error KEEP does.
        if result == PrefilterResult.DROP:
            decision = GCDecision(
                span_id=span.span_id,
                tier=Tier.COLD,
                treatment=Treatment.DROP,
                reason="prefilter_drop",
                used_jev=False,
            )
        elif result == PrefilterResult.KEEP and not span.is_error:
            decision = GCDecision(
                span_id=span.span_id,
                tier=Tier.HOT,
                treatment=Treatment.INCLUDE_FULL,
                reason="prefilter_keep",
                used_jev=False,
            )
        else:
            # `span_id` is already unique per span, so it doubles as the
            # scorer's `item_id` -- no need for `new_item_id()` here.
            jev_result = await self._scorer.submit(
                span, item_id=span.span_id, task_context=self._current_task
            )
            decision = self._policy.decide(span, jev_result)

        item = build_context_item(span, decision)
        self._store.put(item)
        self._telemetry.record_decision(decision, item.token_count)
        return decision

    def pin(self, span_id: str) -> None:
        """Marks a span as pinned (system prompt, explicit user constraint,
        open TODO) -- the prefilter always keeps pinned spans."""
        self._pinned_span_ids.add(span_id)

    def mark_referenced(self, span_id: str) -> None:
        """Marks a span as referenced by a later turn -- the prefilter keeps
        referenced spans, and any WARM/COLD item already stored for it is
        promoted back to HOT."""
        self._referenced_span_ids.add(span_id)
        self._store.promote_if_referenced(span_id)

    def advance_turn(self, turn_index: int | None = None) -> None:
        self._turn_index = turn_index if turn_index is not None else self._turn_index + 1

    def build_context(self, *, task: str, budget_tokens: int) -> str:
        """Assembles the next prompt's context string from everything
        currently HOT or WARM in the store (COLD items are archived and
        never auto-reinstated), greedily filling `budget_tokens` by
        relevance with errored spans always winning a slot first."""
        self._current_task = task
        candidates = self._store.list_by_tier(Tier.HOT) + self._store.list_by_tier(Tier.WARM)
        selected = BudgetAllocator.allocate(candidates, budget_tokens)
        return assemble_context(selected)

    def stats(self) -> GCStats:
        return self._telemetry.snapshot()

    async def wait_all(self) -> None:
        """Awaits every span dispatched via `attach_to_tracer_provider` that
        hasn't finished flowing through `observe` yet. No-op if
        `attach_to_tracer_provider` was never called (e.g. manual-observe-only
        hosts, which already `await` each `observe` call directly)."""
        if self._processor is not None:
            await self._processor.wait_all()

    async def aclose(self) -> None:
        if self._owns_client and isinstance(self._client, HTTPJevClient):
            await self._client.aclose()
