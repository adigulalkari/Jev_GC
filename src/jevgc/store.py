"""TieredContextStore + Backend protocol (SPEC.md §4.6).

`TieredContextStore` is the only module allowed to change an item's `Tier`
after the policy has decided it -- e.g. re-promoting a WARM item to HOT
because a later span referenced its `span_id`. That's a deterministic rule
(referenced => keep close at hand), never a Jev call.

Two responsibilities live here beyond plain tiering, both serving the same
question -- "what did we evict, and can we get it back?":

* an append-only **eviction log** of every tier transition, so a session's
  eviction decisions can be replayed and audited after the fact (see
  `regret.py`), and
* an **archive** of immutable full-content snapshots, because a compressed
  `ContextItem`'s `rendered_text` is already a pointer or a summary by the
  time it leaves HOT. Without the archive, promoting an item back to HOT
  restores a pointer, not content -- the tier moves and nothing is actually
  rehydrated.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from jevgc.archive import Archive, ColdIndexEntry, EvictionEvent
from jevgc.context_builder import estimate_tokens
from jevgc.models import ContextItem, ErrorTreatment, Tier, Treatment, now_unix_ns

logger = logging.getLogger(__name__)


class Backend(Protocol):
    def get(self, span_id: str) -> ContextItem | None: ...
    def put(self, item: ContextItem) -> None: ...
    def list_by_tier(self, tier: Tier) -> list[ContextItem]: ...
    def move_tier(self, span_id: str, tier: Tier) -> None: ...


class TieredContextStore:
    def __init__(self, backend: Backend, archive: Archive | None = None) -> None:
        self._backend = backend
        self._archive = archive if archive is not None else Archive()
        self._eviction_log: list[EvictionEvent] = []

    @property
    def archive(self) -> Archive:
        return self._archive

    def eviction_log(self) -> Sequence[EvictionEvent]:
        """Every tier transition this store has seen, oldest first. Append-only
        and never pruned: the point of keeping it is that a false eviction is
        only visible in hindsight, several turns after the decision."""
        return tuple(self._eviction_log)

    def put(
        self,
        item: ContextItem,
        *,
        full_text: str | None = None,
        turn_index: int = 0,
    ) -> None:
        """Stores `item`. `full_text` is the span's uncompressed content --
        pass it so an item leaving HOT can be rehydrated later; without it,
        a compressed item's original content is unrecoverable."""
        previous = self._backend.get(item.span_id)
        from_tier = previous.tier if previous is not None else None
        self._backend.put(item)

        if full_text is not None and item.tier != Tier.HOT:
            self._archive.record(item.span_id, full_text, item.tier, turn_index)

        born_hot = from_tier is None and item.tier == Tier.HOT
        if from_tier != item.tier and not born_hot:
            self._record_transition(
                item.span_id, from_tier, item.tier, turn_index, item.decision.reason
            )

    def get(self, span_id: str) -> ContextItem | None:
        return self._backend.get(span_id)

    def list_by_tier(self, tier: Tier) -> list[ContextItem]:
        return self._backend.list_by_tier(tier)

    def move_tier(
        self, span_id: str, tier: Tier, *, turn_index: int = 0, reason: str = "move_tier"
    ) -> None:
        item = self._backend.get(span_id)
        from_tier = item.tier if item is not None else None
        self._backend.move_tier(span_id, tier)
        if from_tier != tier:
            self._record_transition(span_id, from_tier, tier, turn_index, reason)

    def promote_if_referenced(self, span_id: str) -> bool:
        """Re-promote `span_id` to HOT if it's currently WARM/COLD. Returns
        True if a promotion happened. No-op (returns False) if the item
        isn't tracked or is already HOT.

        Moves the tier only. Prefer `rehydrate`, which also restores the
        item's full content from the archive -- a promoted pointer is still
        a pointer.
        """
        item = self._backend.get(span_id)
        if item is None or item.tier == Tier.HOT:
            return False
        self._backend.move_tier(span_id, Tier.HOT)
        self._record_transition(span_id, item.tier, Tier.HOT, 0, "promote_if_referenced")
        return True

    def rehydrate(
        self, span_id: str, *, turn_index: int = 0, reason: str = "rehydrated"
    ) -> ContextItem | None:
        """Bring `span_id` back to HOT with its full content restored from the
        archive, not just its tier flipped.

        Returns the live item, or None if the span isn't tracked at all. An
        item already HOT is returned unchanged -- rehydration is idempotent so
        a caller can ask for it without first checking the current tier.
        """
        item = self._backend.get(span_id)
        if item is None:
            return None
        if item.tier == Tier.HOT:
            return item

        archived = self._archive.get(span_id)
        if archived is None or archived.full_text is None:
            # Nothing to restore -- either the span was stored without its full
            # text (e.g. a backend rehydrated from disk with no matching
            # archive), or its content was released under the archive's memory
            # budget. Promote anyway rather than refusing: a pointer in HOT is
            # worse than content, but better than an item the agent can't reach.
            logger.warning(
                "rehydrate(%s): %s, promoting tier only",
                span_id,
                "content released under memory pressure"
                if archived is not None
                else "no archived snapshot",
            )
            self._backend.move_tier(span_id, Tier.HOT)
            rehydrated = self._backend.get(span_id)
        else:
            treatment: Treatment | ErrorTreatment = (
                ErrorTreatment.KEEP_FULL_TRACE
                if isinstance(item.decision.treatment, ErrorTreatment)
                else Treatment.INCLUDE_FULL
            )
            rehydrated = ContextItem(
                span_id=span_id,
                tier=Tier.HOT,
                rendered_text=archived.full_text,
                token_count=estimate_tokens(archived.full_text),
                decision=item.decision.model_copy(
                    update={"tier": Tier.HOT, "treatment": treatment, "reason": reason}
                ),
            )
            self._backend.put(rehydrated)

        self._record_transition(span_id, item.tier, Tier.HOT, turn_index, reason)
        return rehydrated

    def cold_index(self) -> list[ColdIndexEntry]:
        """Keyword-only index of everything currently evicted (WARM or COLD).

        This is what makes eviction recoverable rather than terminal: an agent
        can't ask for a `span_id` it has never seen, so something addressable
        has to stay visible. Carries keywords, never content -- cheap enough to
        sit in every prompt without undoing the eviction it describes.
        """
        return self._archive.cold_index(self._tier_of)

    def search_cold(self, query: str, limit: int = 5) -> list[ColdIndexEntry]:
        """Find evicted spans whose archived keywords overlap `query`, so a
        later task can discover context it didn't know existed."""
        return self._archive.search(query, self._tier_of, limit=limit)

    def _tier_of(self, span_id: str) -> Tier | None:
        item = self._backend.get(span_id)
        return item.tier if item is not None else None

    def _record_transition(
        self,
        span_id: str,
        from_tier: Tier | None,
        to_tier: Tier,
        turn_index: int,
        reason: str,
    ) -> None:
        self._eviction_log.append(
            EvictionEvent(
                span_id=span_id,
                from_tier=from_tier,
                to_tier=to_tier,
                turn_index=turn_index,
                reason=reason,
                timestamp_unix_ns=now_unix_ns(),
            )
        )
