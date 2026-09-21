"""TieredContextStore + Backend protocol (SPEC.md §4.6).

`TieredContextStore` is the only module allowed to change an item's `Tier`
after the policy has decided it -- e.g. re-promoting a WARM item to HOT
because a later span referenced its `span_id`. That's a deterministic rule
(referenced => keep close at hand), never a Jev call.
"""

from __future__ import annotations

from typing import Protocol

from jevgc.models import ContextItem, Tier


class Backend(Protocol):
    def get(self, span_id: str) -> ContextItem | None: ...
    def put(self, item: ContextItem) -> None: ...
    def list_by_tier(self, tier: Tier) -> list[ContextItem]: ...
    def move_tier(self, span_id: str, tier: Tier) -> None: ...


class TieredContextStore:
    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def put(self, item: ContextItem) -> None:
        self._backend.put(item)

    def get(self, span_id: str) -> ContextItem | None:
        return self._backend.get(span_id)

    def list_by_tier(self, tier: Tier) -> list[ContextItem]:
        return self._backend.list_by_tier(tier)

    def move_tier(self, span_id: str, tier: Tier) -> None:
        self._backend.move_tier(span_id, tier)

    def promote_if_referenced(self, span_id: str) -> bool:
        """Re-promote `span_id` to HOT if it's currently WARM/COLD. Returns
        True if a promotion happened. No-op (returns False) if the item
        isn't tracked or is already HOT."""
        item = self._backend.get(span_id)
        if item is None or item.tier == Tier.HOT:
            return False
        self._backend.move_tier(span_id, Tier.HOT)
        return True
