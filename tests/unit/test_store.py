from __future__ import annotations

import pytest

from jevgc.backends.memory import InMemoryBackend
from jevgc.exceptions import StoreError
from jevgc.models import ContextItem, GCDecision, Tier, Treatment
from jevgc.store import TieredContextStore


def _item(span_id: str, tier: Tier) -> ContextItem:
    return ContextItem(
        span_id=span_id,
        tier=tier,
        rendered_text="hello",
        token_count=1,
        decision=GCDecision(
            span_id=span_id, tier=tier, treatment=Treatment.INCLUDE_FULL, reason="test", used_jev=False
        ),
    )


def test_in_memory_backend_crud():
    backend = InMemoryBackend()
    item = _item("s1", Tier.HOT)
    backend.put(item)

    assert backend.get("s1") == item
    assert backend.get("missing") is None
    assert backend.list_by_tier(Tier.HOT) == [item]
    assert backend.list_by_tier(Tier.WARM) == []


def test_in_memory_backend_move_tier():
    backend = InMemoryBackend()
    backend.put(_item("s1", Tier.HOT))
    backend.move_tier("s1", Tier.WARM)

    assert backend.get("s1").tier == Tier.WARM
    assert backend.list_by_tier(Tier.HOT) == []
    assert len(backend.list_by_tier(Tier.WARM)) == 1


def test_move_tier_unknown_span_raises_store_error():
    backend = InMemoryBackend()
    with pytest.raises(StoreError):
        backend.move_tier("nope", Tier.HOT)


def test_promote_if_referenced_moves_warm_to_hot():
    store = TieredContextStore(InMemoryBackend())
    store.put(_item("s1", Tier.WARM))

    promoted = store.promote_if_referenced("s1")

    assert promoted is True
    assert store.get("s1").tier == Tier.HOT


def test_promote_if_referenced_is_noop_when_already_hot():
    store = TieredContextStore(InMemoryBackend())
    store.put(_item("s1", Tier.HOT))

    assert store.promote_if_referenced("s1") is False


def test_promote_if_referenced_is_noop_for_unknown_span():
    store = TieredContextStore(InMemoryBackend())
    assert store.promote_if_referenced("nope") is False
