"""Immutable content archive + an always-visible keyword index over it.

WHY THESE TYPES LIVE HERE AND NOT IN `models.py`
------------------------------------------------
SPEC.md declares `models.py` a frozen contract: it holds the types that cross
the OTel -> prefilter -> scorer -> policy -> store pipeline boundaries listed
in SPEC.md §3.1. `EvictionEvent`, `ArchiveEntry` and `ColdIndexEntry` are not
part of that pipeline -- they belong to a new subsystem (discovery and
rehydration of already-evicted spans) that sits beside it. Adding them to the
frozen contract would widen it for types no pipeline stage passes, so they
live in their own module instead. That is the documented, deliberate reason
SPEC.md §1 asks for.

WHY THE SUBSYSTEM EXISTS
------------------------
SPEC.md §2 principle 3 promises nothing is ever hard-deleted, but a span
demoted to WARM/COLD was in practice a dead end: `build_context()` only draws
on HOT/WARM, COLD is never auto-reinstated, and re-promoting requires already
knowing the `span_id` -- which an agent that can no longer see the span cannot
know. Worse, a compressed span's `ContextItem.rendered_text` is *already* a
pointer or summary, so even a successful promotion restores the pointer rather
than the content. This module fixes both halves: it keeps the full text that
jev-gc observed at eviction time, and it publishes a cheap keyword index the
agent can always see, so it can name what it wants back.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, OrderedDict
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from jevgc.models import Tier, now_unix_ns

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")

_MIN_TOKEN_LENGTH = 3

#: Default archive content budget. 8 MiB is far more than a typical session
#: produces (the codebase_triage_agent example archives well under 100 KB),
#: so the common case never releases anything -- the budget exists to put a
#: ceiling on the pathological case, not to shape normal behavior.
DEFAULT_MAX_CONTENT_BYTES = 8 * 1024 * 1024

# Deliberately tiny and hand-written: SPEC.md §2 principle 1 says code
# calculates. A stopword list is a threshold-like judgment, so it stays in
# code -- no ML, no embeddings, no model download, nothing that could make
# keyword extraction non-reproducible between two runs of the same test.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
        "had", "has", "have", "her", "his", "its", "our", "out", "was", "were",
        "who", "why", "how", "with", "this", "that", "these", "those", "from",
        "into", "onto", "than", "then", "them", "they", "there", "their",
        "been", "being", "does", "did", "done", "will", "would", "could",
        "should", "shall", "must", "may", "might", "about", "above", "after",
        "again", "also", "because", "before", "below", "between", "both",
        "during", "each", "here", "just", "more", "most", "only", "other",
        "over", "same", "some", "such", "under", "until", "very", "when",
        "where", "which", "while", "your", "yours", "what", "whom", "own",
        "too", "off", "one", "two",
    }
)


class EvictionEvent(BaseModel):
    """One recorded tier transition, kept so the history of *why* a span left
    the prompt survives the transition itself.

    A `GCDecision` says what was decided now; a trail of `EvictionEvent`s says
    how an item got to where it is, which is what an audit (or a regret pass
    deciding whether an eviction was a mistake) actually needs.
    """

    model_config = ConfigDict(frozen=True)

    span_id: str
    from_tier: Tier | None  # None == this was the span's first tier assignment
    to_tier: Tier
    turn_index: int
    reason: str  # GCDecision.reason, or a rehydration reason string
    timestamp_unix_ns: int


class ArchiveEntry(BaseModel):
    """A snapshot of one span's full content, taken when it was archived.

    `tier_at_archive` is the tier at snapshot time and is deliberately *not*
    kept in sync with the span's current tier -- it is part of the historical
    record. Anything that needs the live tier asks the store (see
    `Archive.cold_index`).

    `full_text` is `None` for an entry whose content was released under the
    archive's memory budget. The rest of the entry survives that release --
    crucially the keywords -- so a released span stays discoverable in the
    cold index and still counts in a regret pass; only the ability to restore
    its content verbatim is lost. See `Archive` for why that tradeoff is the
    one worth making.
    """

    model_config = ConfigDict(frozen=True)

    span_id: str
    full_text: str | None
    keywords: tuple[str, ...]
    tier_at_archive: Tier
    turn_index: int
    timestamp_unix_ns: int

    @property
    def has_content(self) -> bool:
        return self.full_text is not None


class ArchiveStats(BaseModel):
    """What the archive is currently costing, in the units a host tuning
    `max_content_bytes` actually needs."""

    model_config = ConfigDict(frozen=True)

    entry_count: int  # archived spans; metadata + keywords, always retained
    retained_bytes: int  # bytes of `full_text` currently held in memory
    released_count: int  # entries whose content was dropped under pressure


class ColdIndexEntry(BaseModel):
    """One advertised line of the always-visible index: enough for an agent to
    recognize that something relevant exists and ask for it by `span_id`, and
    deliberately nothing more.

    Carrying `full_text` here would quietly undo the eviction it describes --
    the index is in *every* prompt, so its per-entry cost is what keeps the
    whole mechanism affordable.
    """

    model_config = ConfigDict(frozen=True)

    span_id: str
    tier: Tier  # the span's CURRENT tier, not tier_at_archive
    keywords: tuple[str, ...]
    turn_index: int


def extract_keywords(text: str, max_keywords: int = 12) -> tuple[str, ...]:
    """Pick the tokens that best identify `text`, deterministically.

    Deliberately stdlib-only frequency ranking rather than embeddings: SPEC.md
    §2 principle 1 reserves probabilistic judgment for Jev, and §2 principle 7
    requires reproducible output for the same input. An embedding model would
    make the index un-auditable and add a dependency jev-gc does not need to
    answer "does the agent recognize this string?".

    Ties are broken by token length (longer first -- more specific) and then
    alphabetically, so the result is stable regardless of token order in the
    source text.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    counts = Counter(
        t for t in tokens if len(t) > _MIN_TOKEN_LENGTH - 1 and t not in _STOPWORDS
    )
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return tuple(token for token, _ in ranked[:max_keywords])


def _content_bytes(full_text: str) -> int:
    """Budget in encoded bytes, not `len()`: a session of CJK or emoji-heavy
    tool output costs several times its character count in memory."""
    return len(full_text.encode("utf-8"))


def _query_tokens(query: str) -> set[str]:
    """Tokenize a search query exactly as `extract_keywords` tokenizes content
    -- but without the `max_keywords` truncation, since a long query is the
    caller telling us more about what they want, not noise to trim."""
    return {
        t.lower()
        for t in _TOKEN_RE.findall(query)
        if len(t) > _MIN_TOKEN_LENGTH - 1 and t.lower() not in _STOPWORDS
    }


class Archive:
    """Write-once store of the full text of spans jev-gc has observed, held
    under a fixed memory budget.

    `record` is write-once by design: once a `span_id` is archived, a later
    `record` for the same id is a no-op. An archived pointer must resolve to an
    immutable snapshot of what jev-gc actually observed, never to a live object
    that could drift between eviction and retrieval -- otherwise "restore this
    span" would hand the agent content that never existed at the moment the
    eviction decision was made, and every audit of that decision would be
    reading different evidence than the decision saw.

    WHY THERE IS A BUDGET AT ALL
    ----------------------------
    An unbounded archive quietly inverts the point of the library. jev-gc
    exists to cut what a long session sends to the model, and retaining every
    evicted span's full text forever would mean a process that trims its
    prompts while its own memory grows without limit -- worst exactly in the
    long-running sessions the library is for.

    WHAT GIVES WAY UNDER PRESSURE
    -----------------------------
    Content, never discoverability. When `max_content_bytes` would be
    exceeded, the least-recently-used entries' `full_text` is released while
    their keywords, tier and turn are kept. A released span still appears in
    `cold_index`, still answers `search`, and still participates in a regret
    pass; what it can no longer do is hand back its exact bytes. Keywords cost
    on the order of a hundred bytes against content that can run to kilobytes,
    so this keeps the mechanism's *visible* half affordable almost indefinitely
    and spends the budget only on the half that can be reconstructed by
    re-running the tool.

    This is the one place jev-gc knowingly bends SPEC.md §2 principle 3
    ("never hard-delete"). It is bounded and explicit rather than silent: the
    caller sets the budget, `stats()` reports what was given up, and
    `TieredContextStore.rehydrate` degrades to a tier promotion with a logged
    warning rather than failing. Set `max_content_bytes` high enough to cover
    a session and nothing is released at all.

    Not thread-safe by intent: jev-gc's pipeline is single-writer per `JevGC`
    instance (SPEC.md §6 forbids global mutable state), so a lock here would
    buy nothing but contention.
    """

    def __init__(self, max_content_bytes: int = DEFAULT_MAX_CONTENT_BYTES) -> None:
        if max_content_bytes < 0:
            raise ValueError("max_content_bytes must be >= 0")
        self._max_content_bytes = max_content_bytes
        # dict preserves insertion order, which is what makes `entries()`
        # deterministic without a separate sort key.
        self._entries: dict[str, ArchiveEntry] = {}
        # Recency is tracked separately rather than by reordering `_entries`,
        # so that use does not perturb the historical order `entries()` reports.
        self._recency: OrderedDict[str, None] = OrderedDict()
        self._retained_bytes = 0
        self._released_count = 0

    def record(self, span_id: str, full_text: str, tier: Tier, turn_index: int) -> None:
        """Archive `full_text` for `span_id`, unless it is already archived.

        The no-op-on-repeat branch is the headline property, not an
        optimization -- see the class docstring.
        """
        if span_id in self._entries:
            logger.debug(
                "archive: span_id=%s already archived; keeping original snapshot", span_id
            )
            return
        self._entries[span_id] = ArchiveEntry(
            span_id=span_id,
            full_text=full_text,
            keywords=extract_keywords(full_text),
            tier_at_archive=tier,
            turn_index=turn_index,
            timestamp_unix_ns=now_unix_ns(),
        )
        self._recency[span_id] = None
        self._retained_bytes += _content_bytes(full_text)
        self._enforce_budget()

    def get(self, span_id: str) -> ArchiveEntry | None:
        entry = self._entries.get(span_id)
        if entry is not None and entry.has_content:
            # Reading content is what makes an entry worth keeping: a span the
            # agent keeps rehydrating should outlive one nothing has asked for.
            self._recency.move_to_end(span_id)
        return entry

    def stats(self) -> ArchiveStats:
        return ArchiveStats(
            entry_count=len(self._entries),
            retained_bytes=self._retained_bytes,
            released_count=self._released_count,
        )

    def _enforce_budget(self) -> None:
        """Release least-recently-used content until the budget is met.

        Enforced strictly, including against the entry just recorded: a bound
        that one oversized span can exceed is not a bound. That case is logged
        at warning level because it means the archive is configured too small
        to ever restore that span.
        """
        for span_id in list(self._recency):
            if self._retained_bytes <= self._max_content_bytes:
                return
            self._release(span_id)

    def _release(self, span_id: str) -> None:
        entry = self._entries[span_id]
        if entry.full_text is None:
            self._recency.pop(span_id, None)
            return
        self._retained_bytes -= _content_bytes(entry.full_text)
        self._entries[span_id] = entry.model_copy(update={"full_text": None})
        self._recency.pop(span_id, None)
        self._released_count += 1
        logger.warning(
            "archive: released content for span_id=%s to stay within "
            "max_content_bytes=%d; it remains discoverable but can no longer "
            "be rehydrated verbatim",
            span_id,
            self._max_content_bytes,
        )

    def entries(self) -> list[ArchiveEntry]:
        """All entries in insertion order -- stable across runs so tests and
        audits can compare two archives directly."""
        return list(self._entries.values())

    def __contains__(self, span_id: str) -> bool:
        return span_id in self._entries

    def cold_index(self, tier_of: Callable[[str], Tier | None]) -> list[ColdIndexEntry]:
        """Build the index that goes in every prompt: one line per archived
        span the agent can no longer see.

        `tier_of` is injected rather than read off `ArchiveEntry` because the
        archive holds history while the store holds truth -- a span archived
        as WARM may since have been promoted back to HOT, and advertising it
        then would be paying tokens to tell the agent about something already
        in front of it. Spans `tier_of` no longer knows about are skipped too:
        offering to restore an untracked span would be a promise this module
        cannot keep.
        """
        index: list[ColdIndexEntry] = []
        for entry in self._entries.values():
            current = tier_of(entry.span_id)
            if current is None or current == Tier.HOT:
                continue
            index.append(
                ColdIndexEntry(
                    span_id=entry.span_id,
                    tier=current,
                    keywords=entry.keywords,
                    turn_index=entry.turn_index,
                )
            )
        return index

    def search(
        self,
        query: str,
        tier_of: Callable[[str], Tier | None],
        limit: int = 5,
    ) -> list[ColdIndexEntry]:
        """Find evicted spans whose keywords overlap `query`.

        Scoring is plain keyword overlap: the point is discovery, not ranking
        quality. An agent that finds the span it half-remembers can ask for it
        back by `span_id`; a cleverer scorer would add a dependency and a
        non-reproducible result for no extra recall at this scale. Zero-overlap
        entries are excluded rather than ranked last, so an unrelated query
        returns nothing instead of a plausible-looking wrong answer.

        Ties break on `span_id` ascending purely for determinism (§2
        principle 7).
        """
        wanted = _query_tokens(query)
        if not wanted:
            return []

        scored: list[tuple[int, ColdIndexEntry]] = []
        for entry in self.cold_index(tier_of):
            overlap = len(wanted.intersection(entry.keywords))
            if overlap:
                scored.append((overlap, entry))

        scored.sort(key=lambda pair: (-pair[0], pair[1].span_id))
        return [entry for _, entry in scored[:limit]]
