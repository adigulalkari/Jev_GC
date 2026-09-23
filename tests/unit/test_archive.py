from __future__ import annotations

from jevgc.archive import (
    Archive,
    ArchiveEntry,
    ColdIndexEntry,
    EvictionEvent,
    extract_keywords,
)
from jevgc.models import Tier


def _tier_map(mapping: dict[str, Tier]):
    """A stand-in for the store's tier lookup -- the archive only needs a
    callable, so no mock/backend is required here."""
    return lambda span_id: mapping.get(span_id)


# --------------------------------------------------------------------------
# extract_keywords
# --------------------------------------------------------------------------


def test_extract_keywords_is_deterministic_for_same_input():
    text = "warehouse shipment routing warehouse invoice carrier shipment warehouse"
    assert extract_keywords(text) == extract_keywords(text)


def test_extract_keywords_ranks_by_frequency_then_length_then_alpha():
    # warehouse x3, shipment x2, then two singletons of equal frequency:
    # "carrier" (7 chars) beats "routing" (7 chars) alphabetically.
    text = "warehouse shipment routing warehouse carrier shipment warehouse"
    assert extract_keywords(text) == ("warehouse", "shipment", "carrier", "routing")


def test_extract_keywords_longer_token_wins_equal_frequency_tie():
    # Equal frequency (1 each) -> longer token first, despite alphabetics.
    assert extract_keywords("zzz invoices") == ("invoices", "zzz")


def test_extract_keywords_drops_stopwords_and_short_tokens():
    keywords = extract_keywords("the ERP is a of table with an id and no rows")
    assert "the" not in keywords
    assert "with" not in keywords
    assert "is" not in keywords
    assert "id" not in keywords
    assert "erp" in keywords
    assert "table" in keywords


def test_extract_keywords_lowercases_and_splits_on_word_chars():
    assert extract_keywords("Query_ERP-Table!") == ("query_erp", "table")


def test_extract_keywords_respects_max_keywords():
    text = " ".join(f"token{i}" for i in range(50))
    assert len(extract_keywords(text, max_keywords=5)) == 5


def test_extract_keywords_on_empty_text():
    assert extract_keywords("") == ()


# --------------------------------------------------------------------------
# record / get / entries / __contains__
# --------------------------------------------------------------------------


def test_record_stores_full_text_and_derived_keywords():
    archive = Archive()
    archive.record("s1", "warehouse shipment delayed", Tier.WARM, turn_index=3)

    entry = archive.get("s1")
    assert isinstance(entry, ArchiveEntry)
    assert entry.span_id == "s1"
    assert entry.full_text == "warehouse shipment delayed"
    assert entry.tier_at_archive == Tier.WARM
    assert entry.turn_index == 3
    assert "warehouse" in entry.keywords
    assert entry.timestamp_unix_ns > 0


def test_record_is_write_once_and_does_not_overwrite():
    archive = Archive()
    archive.record("s1", "original observed content", Tier.WARM, turn_index=1)
    first = archive.get("s1")

    archive.record("s1", "DIFFERENT later content", Tier.COLD, turn_index=9)

    entry = archive.get("s1")
    assert entry == first
    assert entry.full_text == "original observed content"
    assert entry.tier_at_archive == Tier.WARM
    assert entry.turn_index == 1
    assert len(archive.entries()) == 1


def test_get_returns_none_for_unknown_span():
    assert Archive().get("nope") is None


def test_contains_reflects_what_has_been_recorded():
    archive = Archive()
    assert "s1" not in archive
    archive.record("s1", "content", Tier.WARM, turn_index=0)
    assert "s1" in archive


def test_entries_preserves_insertion_order():
    archive = Archive()
    for span_id in ("s3", "s1", "s2"):
        archive.record(span_id, f"content for {span_id}", Tier.WARM, turn_index=0)

    assert [e.span_id for e in archive.entries()] == ["s3", "s1", "s2"]


def test_archive_entry_is_frozen():
    archive = Archive()
    archive.record("s1", "content", Tier.WARM, turn_index=0)
    entry = archive.get("s1")
    try:
        entry.full_text = "tampered"
    except Exception:
        return
    raise AssertionError("ArchiveEntry should be immutable")


# --------------------------------------------------------------------------
# cold_index
# --------------------------------------------------------------------------


def test_cold_index_skips_hot_and_untracked_spans():
    archive = Archive()
    archive.record("hot", "erp table routing", Tier.WARM, turn_index=1)
    archive.record("warm", "warehouse shipment", Tier.WARM, turn_index=2)
    archive.record("gone", "forgotten content", Tier.COLD, turn_index=3)

    index = archive.cold_index(_tier_map({"hot": Tier.HOT, "warm": Tier.WARM}))

    assert [e.span_id for e in index] == ["warm"]
    assert isinstance(index[0], ColdIndexEntry)


def test_cold_index_reports_current_tier_not_tier_at_archive():
    archive = Archive()
    archive.record("s1", "warehouse shipment", Tier.WARM, turn_index=2)

    index = archive.cold_index(_tier_map({"s1": Tier.COLD}))

    assert index[0].tier == Tier.COLD
    assert archive.get("s1").tier_at_archive == Tier.WARM


def test_cold_index_never_exposes_full_text():
    archive = Archive()
    archive.record("s1", "secret payload body", Tier.COLD, turn_index=1)

    entry = archive.cold_index(_tier_map({"s1": Tier.COLD}))[0]

    assert not hasattr(entry, "full_text")
    assert "secret" in entry.keywords  # keywords only, no body


def test_cold_index_is_empty_for_empty_archive():
    assert Archive().cold_index(_tier_map({})) == []


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def _seeded_archive():
    archive = Archive()
    archive.record("s1", "warehouse shipment carrier delay", Tier.WARM, turn_index=1)
    archive.record("s2", "warehouse inventory count", Tier.COLD, turn_index=2)
    archive.record("s3", "invoice payment terms", Tier.WARM, turn_index=3)
    return archive


def test_search_ranks_by_keyword_overlap_descending():
    archive = _seeded_archive()
    tier_of = _tier_map({"s1": Tier.WARM, "s2": Tier.COLD, "s3": Tier.WARM})

    results = archive.search("warehouse shipment", tier_of)

    assert [e.span_id for e in results] == ["s1", "s2"]  # 2 overlaps, then 1


def test_search_excludes_zero_overlap_entries():
    archive = _seeded_archive()
    tier_of = _tier_map({"s1": Tier.WARM, "s2": Tier.COLD, "s3": Tier.WARM})

    assert archive.search("quantum astrophysics", tier_of) == []


def test_search_tie_break_is_span_id_ascending():
    archive = Archive()
    for span_id in ("s9", "s2", "s5"):
        archive.record(span_id, "warehouse content", Tier.WARM, turn_index=0)
    tier_of = _tier_map({"s9": Tier.WARM, "s2": Tier.WARM, "s5": Tier.WARM})

    results = archive.search("warehouse", tier_of)

    assert [e.span_id for e in results] == ["s2", "s5", "s9"]


def test_search_respects_limit():
    archive = Archive()
    for i in range(10):
        archive.record(f"s{i}", "warehouse content", Tier.WARM, turn_index=0)
    tier_of = _tier_map({f"s{i}": Tier.WARM for i in range(10)})

    assert len(archive.search("warehouse", tier_of, limit=3)) == 3


def test_search_skips_hot_spans_like_cold_index_does():
    archive = _seeded_archive()
    tier_of = _tier_map({"s1": Tier.HOT, "s2": Tier.COLD, "s3": Tier.WARM})

    results = archive.search("warehouse", tier_of)

    assert [e.span_id for e in results] == ["s2"]


def test_search_with_only_stopwords_returns_nothing():
    archive = _seeded_archive()
    tier_of = _tier_map({"s1": Tier.WARM, "s2": Tier.COLD, "s3": Tier.WARM})

    assert archive.search("the and for with", tier_of) == []


def test_search_does_not_truncate_long_queries():
    # A query with more than max_keywords (12) meaningful tokens must still
    # match on a term that would have been truncated away by extract_keywords.
    archive = Archive()
    archive.record("s1", "peculiar", Tier.COLD, turn_index=0)
    tier_of = _tier_map({"s1": Tier.COLD})
    query = " ".join([f"filler{i}" for i in range(20)] + ["peculiar"])

    assert [e.span_id for e in archive.search(query, tier_of)] == ["s1"]


# --------------------------------------------------------------------------
# EvictionEvent
# --------------------------------------------------------------------------


def test_eviction_event_allows_none_from_tier_for_first_assignment():
    event = EvictionEvent(
        span_id="s1",
        from_tier=None,
        to_tier=Tier.HOT,
        turn_index=0,
        reason="initial_assignment",
        timestamp_unix_ns=123,
    )

    assert event.from_tier is None
    assert event.to_tier == Tier.HOT


def test_eviction_event_is_frozen():
    event = EvictionEvent(
        span_id="s1",
        from_tier=Tier.HOT,
        to_tier=Tier.WARM,
        turn_index=2,
        reason="low_relevance",
        timestamp_unix_ns=123,
    )
    try:
        event.to_tier = Tier.COLD
    except Exception:
        return
    raise AssertionError("EvictionEvent should be immutable")


# --------------------------------------------------------------------------
# memory budget
# --------------------------------------------------------------------------


def test_stats_account_for_retained_content_in_utf8_bytes():
    archive = Archive()
    archive.record("s1", "café", Tier.COLD, 0)  # 5 bytes in utf-8, 4 chars

    stats = archive.stats()
    assert (stats.entry_count, stats.retained_bytes, stats.released_count) == (1, 5, 0)


def test_nothing_is_released_while_under_budget():
    archive = Archive(max_content_bytes=1000)
    archive.record("s1", "x" * 100, Tier.COLD, 0)
    archive.record("s2", "y" * 100, Tier.WARM, 1)

    assert archive.stats().released_count == 0
    assert archive.get("s1").full_text == "x" * 100


def test_over_budget_releases_least_recently_used_content_first():
    archive = Archive(max_content_bytes=250)
    archive.record("old", "o" * 100, Tier.COLD, 0)
    archive.record("mid", "m" * 100, Tier.COLD, 1)
    archive.record("new", "n" * 100, Tier.COLD, 2)  # 300 > 250 -> release oldest

    assert archive.get("old").full_text is None
    assert archive.get("mid").full_text == "m" * 100
    assert archive.get("new").full_text == "n" * 100
    stats = archive.stats()
    assert (stats.retained_bytes, stats.released_count, stats.entry_count) == (200, 1, 3)


def test_released_span_keeps_its_keywords_and_stays_discoverable():
    """The point of releasing content rather than the whole entry: eviction
    must not become undiscoverable just because memory got tight."""
    archive = Archive(max_content_bytes=60)
    archive.record("s1", "warehouse shipment manifest rotterdam depot", Tier.COLD, 0)
    archive.record("s2", "invoice reconciliation erp totals ledger", Tier.COLD, 1)

    released = archive.get("s1")
    assert released.full_text is None
    assert "rotterdam" in released.keywords

    tier_of = _tier_map({"s1": Tier.COLD, "s2": Tier.COLD})
    assert [e.span_id for e in archive.search("rotterdam depot", tier_of)] == ["s1"]
    assert {e.span_id for e in archive.cold_index(tier_of)} == {"s1", "s2"}


def test_reading_content_protects_an_entry_from_the_next_release():
    archive = Archive(max_content_bytes=250)
    archive.record("a", "a" * 100, Tier.COLD, 0)
    archive.record("b", "b" * 100, Tier.COLD, 1)

    archive.get("a")  # refreshes recency, so "b" is now the oldest
    archive.record("c", "c" * 100, Tier.COLD, 2)

    assert archive.get("a").full_text == "a" * 100
    assert archive.get("b").full_text is None


def test_entry_larger_than_the_whole_budget_is_released():
    """A bound one oversized span can exceed is not a bound."""
    archive = Archive(max_content_bytes=10)
    archive.record("huge", "x" * 500, Tier.COLD, 0)

    assert archive.get("huge").full_text is None
    assert archive.stats().retained_bytes == 0
    assert archive.stats().entry_count == 1  # discoverable, just not restorable


def test_reading_does_not_reorder_the_historical_entry_list():
    archive = Archive()
    archive.record("s1", "first", Tier.COLD, 0)
    archive.record("s2", "second", Tier.COLD, 1)

    archive.get("s1")

    assert [e.span_id for e in archive.entries()] == ["s1", "s2"]


def test_negative_budget_is_rejected():
    try:
        Archive(max_content_bytes=-1)
    except ValueError:
        return
    raise AssertionError("Archive should reject a negative budget")
