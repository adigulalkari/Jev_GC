"""Shadow/replay regret detection: did we evict something that mattered?

Every eviction policy in this library is a bet. A span gets demoted to WARM or
COLD because it looks irrelevant *at that turn*, and three turns later it turns
out the agent needed it. Nothing in the normal pipeline can notice that: the
agent simply answers slightly worse, with no error and no signal. This module
is the only place jev-gc looks back and asks whether a past eviction was a
mistake.

The method is a shadow replay the *host* runs, not this library:

1. Run the session once with jev-gc evicting -- the "evicted" run.
2. Run the same session again with no eviction at all (full context) -- the
   "baseline" run.
3. Hand both final output strings, the eviction log, and the archive to
   :func:`find_regret`.

If content that jev-gc evicted visibly resurfaces in the baseline run's output
but not in the evicted run's output, that is evidence -- not proof -- that the
eviction cost the agent something.

**What this actually measures, stated plainly:** keyword overlap. A
`RegretFinding` means "distinctive terms from this evicted span appear in the
full-context answer and are absent from the GC'd answer." It is correlational.
The two runs may diverge for reasons that have nothing to do with the eviction
(LLM sampling nondeterminism, tool responses that changed between runs, a
keyword that is simply a common word). Equally, it under-reports: an eviction
can degrade reasoning without changing which nouns appear in the output. Treat
findings as a ranked list of evictions worth inspecting and as a dial for
tuning `relevance_keep_threshold` / `min_confidence_to_drop` -- never as a
causal claim, and never as a regression gate on its own.

`min_keyword_hits` is the precision/recall knob. Raise it (3-4) for a short
list of high-confidence suspects; lower it to 1 to see every faint signal,
accepting that single-keyword coincidences dominate at that setting.

Pure, synchronous, stdlib + pydantic only: no I/O, no network, no Jev call,
and -- per SPEC.md §1 non-goals -- no LLM call. The host produces the two
output strings; this module only diffs them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

from jevgc.archive import Archive, ArchiveEntry, EvictionEvent
from jevgc.models import Tier

logger = logging.getLogger(__name__)

_EVICTION_TIERS = frozenset({Tier.WARM, Tier.COLD})


class RegretFinding(BaseModel):
    """One evicted span whose distinctive terms surfaced only in the baseline run.

    Exists as a typed record rather than a tuple because these get logged,
    serialized into eval reports, and sorted -- and because `regret_score`
    alone is meaningless without the `eviction_reason` that produced it: the
    reason string is what a maintainer actually tunes against.
    """

    model_config = ConfigDict(frozen=True)

    span_id: str
    evicted_to: Tier
    eviction_reason: str
    turn_index: int
    matched_keywords: tuple[str, ...]
    regret_score: float = Field(ge=0.0, le=1.0)


def find_regret(
    eviction_log: Sequence[EvictionEvent],
    archive: Archive,
    *,
    baseline_output: str,
    evicted_output: str,
    min_keyword_hits: int = 2,
) -> list[RegretFinding]:
    """Diff a shadow replay's two final outputs against what jev-gc evicted.

    Args:
        eviction_log: Every tier transition observed during the evicted run,
            in chronological order. Order in the sequence is authoritative --
            `timestamp_unix_ns` is not consulted, so a caller that reorders
            the log changes the answer.
        archive: Immutable snapshots of evicted spans' content. Spans absent
            from it are skipped: with no recorded text there is nothing to
            look for in either output.
        baseline_output: Final answer from the no-eviction (full context) run.
        evicted_output: Final answer from the run where jev-gc was evicting.
        min_keyword_hits: Minimum distinct keywords that must appear in the
            baseline output and be absent from the evicted output before a
            finding is emitted. Below this, overlap is noise rather than
            evidence -- one shared common noun proves nothing.

    Returns:
        Findings sorted by `regret_score` descending, then `span_id`
        ascending. The secondary key exists so that equal-scoring findings
        come back in a stable order across runs -- these results end up in
        test assertions and diffed eval reports.
    """
    # A span can be evicted, rehydrated, and evicted again. Only its final
    # state decides whether it counts as evicted, so collapse the log to the
    # last event per span first (dicts preserve insertion order, which keeps
    # the downstream walk deterministic even before the explicit sort).
    final_events: dict[str, EvictionEvent] = {}
    for event in eviction_log:
        final_events[event.span_id] = event

    findings: list[RegretFinding] = []
    for span_id, event in final_events.items():
        if event.to_tier not in _EVICTION_TIERS:
            # Last move was back to HOT: the span was rehydrated. That's the
            # success case -- the system recovered -- not a regret.
            logger.debug("span %s ended at %s; not an eviction", span_id, event.to_tier)
            continue

        entry = archive.get(span_id)
        if entry is None:
            logger.debug("span %s evicted but absent from archive; nothing to compare", span_id)
            continue

        matched = _keywords_only_in_baseline(entry, baseline_output, evicted_output)
        if len(matched) < min_keyword_hits:
            logger.debug(
                "span %s: %d keyword hit(s) below min_keyword_hits=%d; treating as noise",
                span_id,
                len(matched),
                min_keyword_hits,
            )
            continue

        findings.append(
            RegretFinding(
                span_id=span_id,
                evicted_to=event.to_tier,
                eviction_reason=event.reason,
                turn_index=event.turn_index,
                matched_keywords=matched,
                regret_score=len(matched) / len(entry.keywords),
            )
        )

    findings.sort(key=lambda f: (-f.regret_score, f.span_id))
    return findings


def _keywords_only_in_baseline(
    entry: ArchiveEntry, baseline_output: str, evicted_output: str
) -> tuple[str, ...]:
    """Keywords present in the baseline output and absent from the evicted one.

    Deduplicates case-insensitively while preserving the archive entry's
    keyword order, so `regret_score` can't be inflated by an entry that
    happens to list the same term twice.
    """
    if not entry.keywords:
        logger.debug("archive entry for span %s has no keywords; nothing to match", entry.span_id)
        return ()

    matched: list[str] = []
    seen: set[str] = set()
    for keyword in entry.keywords:
        folded = keyword.casefold()
        if not folded or folded in seen:
            continue
        seen.add(folded)
        if _contains_token(baseline_output, keyword) and not _contains_token(
            evicted_output, keyword
        ):
            matched.append(keyword)
    return tuple(matched)


@lru_cache(maxsize=512)
def _token_pattern(keyword: str) -> re.Pattern[str]:
    """Whole-token matcher for one keyword.

    Lookarounds rather than `\\b` because keywords are not guaranteed to start
    and end with word characters (`extract_keywords` may emit e.g. `.env`),
    and `\\b`'s meaning flips in that case. The point is that keyword "ok"
    must not match inside "broken".
    """
    return re.compile(
        rf"(?<![0-9A-Za-z_]){re.escape(keyword)}(?![0-9A-Za-z_])",
        re.IGNORECASE,
    )


def _contains_token(text: str, keyword: str) -> bool:
    return _token_pattern(keyword).search(text) is not None
