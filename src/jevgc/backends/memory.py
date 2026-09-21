"""InMemoryBackend: default, dict-based `store.Backend` implementation.
Fine for single-process agents; no persistence across restarts."""

from __future__ import annotations

from jevgc.exceptions import StoreError
from jevgc.models import ContextItem, Tier


class InMemoryBackend:
    def __init__(self) -> None:
        self._items: dict[str, ContextItem] = {}

    def get(self, span_id: str) -> ContextItem | None:
        return self._items.get(span_id)

    def put(self, item: ContextItem) -> None:
        self._items[item.span_id] = item

    def list_by_tier(self, tier: Tier) -> list[ContextItem]:
        return [item for item in self._items.values() if item.tier == tier]

    def move_tier(self, span_id: str, tier: Tier) -> None:
        item = self._items.get(span_id)
        if item is None:
            raise StoreError(f"Cannot move unknown span_id {span_id!r} to tier {tier.value!r}")
        self._items[span_id] = item.model_copy(update={"tier": tier})
