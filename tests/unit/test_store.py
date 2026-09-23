from __future__ import annotations

import pytest

from jevgc.archive import Archive
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


def test_put_logs_eviction_but_not_a_span_born_hot():
    store = TieredContextStore(InMemoryBackend())
    store.put(_item("hot", Tier.HOT))
    store.put(_item("cold", Tier.COLD), full_text="full", turn_index=4)

    log = store.eviction_log()
    assert [(e.span_id, e.from_tier, e.to_tier) for e in log] == [("cold", None, Tier.COLD)]
    assert log[0].turn_index == 4
    assert log[0].reason == "test"


def test_move_tier_logs_transition_with_from_tier():
    store = TieredContextStore(InMemoryBackend())
    store.put(_item("s1", Tier.HOT))
    store.move_tier("s1", Tier.WARM, turn_index=2, reason="budget")

    (event,) = store.eviction_log()
    assert (event.from_tier, event.to_tier, event.reason) == (Tier.HOT, Tier.WARM, "budget")


def test_rehydrate_restores_full_content_not_just_the_tier():
    store = TieredContextStore(InMemoryBackend())
    pointer = _item("s1", Tier.COLD).model_copy(update={"rendered_text": "[tool] -> archived"})
    store.put(pointer, full_text="[tool]\nERP total: $4521.00", turn_index=1)

    item = store.rehydrate("s1", turn_index=9)

    assert item is not None
    assert item.tier == Tier.HOT
    assert "ERP total: $4521.00" in item.rendered_text
    assert item.token_count > pointer.token_count
    assert store.get("s1").rendered_text == item.rendered_text


def test_rehydrate_promotes_tier_only_when_nothing_was_archived():
    store = TieredContextStore(InMemoryBackend())
    store.put(_item("s1", Tier.WARM))  # no full_text -> no archive snapshot

    item = store.rehydrate("s1")

    assert item is not None
    assert item.tier == Tier.HOT
    assert item.rendered_text == "hello"


def test_rehydrate_degrades_to_a_promotion_when_content_was_released():
    """Under memory pressure the archive keeps keywords and drops content, so
    the span is still findable and still promotable -- just not verbatim."""
    store = TieredContextStore(InMemoryBackend(), Archive(max_content_bytes=10))
    pointer = _item("s1", Tier.COLD).model_copy(update={"rendered_text": "[tool] -> archived"})
    store.put(pointer, full_text="warehouse manifest rotterdam " * 20)

    item = store.rehydrate("s1")

    assert item.tier == Tier.HOT
    assert item.rendered_text == "[tool] -> archived"  # content is gone, item is not


def test_rehydrate_is_idempotent_and_returns_none_for_unknown_span():
    store = TieredContextStore(InMemoryBackend())
    store.put(_item("s1", Tier.HOT))

    assert store.rehydrate("s1").tier == Tier.HOT
    assert store.eviction_log() == ()
    assert store.rehydrate("nope") is None


def test_cold_index_covers_evicted_spans_and_drops_them_once_rehydrated():
    store = TieredContextStore(InMemoryBackend())
    store.put(_item("s1", Tier.COLD), full_text="warehouse shipment manifest rotterdam")
    store.put(_item("s2", Tier.HOT), full_text="ignored while hot")

    assert [entry.span_id for entry in store.cold_index()] == ["s1"]
    assert [entry.span_id for entry in store.search_cold("rotterdam manifest")] == ["s1"]
    assert store.search_cold("invoice reconciliation") == []

    store.rehydrate("s1")
    assert store.cold_index() == []
