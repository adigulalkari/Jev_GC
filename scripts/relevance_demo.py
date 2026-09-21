"""Worked comparison: keyword-overlap relevance vs. jev-gc's Jev-backed
relevance scoring, on cases specifically chosen to break keyword matching.

This is not a benchmark -- it's three small, concrete cases for the
"does this actually matter" argument in the main README: situations where
a fixed, deterministic rule (keyword/BM25-style overlap with the task
text) gives the wrong keep/drop answer, and a semantic judgment doesn't.

Run:
    export JEV_API_KEY=...
    python scripts/relevance_demo.py
"""

from __future__ import annotations

import asyncio
import os
import re

from pydantic import SecretStr

from jevgc.jev_client.client import HTTPJevClient, JevBatchRequest

TASK = "Reconcile invoice #4521 totals across the ERP system and warehouse shipment records"

CASES = [
    (
        "shipment_delay",
        "Carrier API: shipment #77 delayed 4 days, held at customs (LAX). No ETA provided.",
        "Shares zero keywords with the task, but is very plausibly *why* an invoice total "
        "doesn't reconcile -- goods invoiced but not yet received.",
    ),
    (
        "unrelated_delay",
        "Carrier API: shipment #12 (customer newsletter mailers) delayed 2 days, printer outage.",
        "Same tool, same shape of output (a delay notice) as the case above -- but for a "
        "shipment that has nothing to do with invoice #4521.",
    ),
    (
        "stale_but_needed_fact",
        "ERP lookup (turn 2): invoice #4521 line items: 40x Unit-A @ $95.00, 22x Unit-B @ $110.55",
        "The literal line-item breakdown behind the total being reconciled -- no shared "
        "keywords with a later turn asking 'does the math check out', but clearly essential.",
    ),
]


def keyword_overlap_score(task: str, content: str) -> float:
    """A simple, deterministic relevance baseline: fraction of the task's
    significant words that also appear in the content. This is the kind of
    rule prefilter.py *could* implement without ever calling Jev."""
    stopwords = {"the", "a", "an", "of", "to", "and", "across", "in", "on", "for"}

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stopwords}

    task_words = words(task)
    content_words = words(content)
    if not task_words:
        return 0.0
    return len(task_words & content_words) / len(task_words)


async def jev_relevance_score(client: HTTPJevClient, content: str) -> float:
    request = JevBatchRequest(
        item_id="demo",
        question_id="relevance",
        type="score",
        instructions="How relevant is this span's content to the current task?",
        state=f"current_task: {TASK}\nspan_content: {content}",
    )
    results = await client.batch_ask([request])
    assert results[0].score is not None
    return results[0].score.score


async def main() -> None:
    api_key = os.environ.get("JEV_API_KEY")
    if not api_key:
        raise SystemExit("Set JEV_API_KEY to run this comparison against the real Jev API.")

    client = HTTPJevClient(api_key=SecretStr(api_key))

    print(f"task: {TASK}\n")
    print(f"{'case':<22} {'keyword-overlap':>16} {'jev relevance':>15}   note")
    print("-" * 100)
    for name, content, note in CASES:
        kw_score = keyword_overlap_score(TASK, content)
        jev_score = await jev_relevance_score(client, content)
        print(f"{name:<22} {kw_score:>16.2f} {jev_score:>15.2f}   {note}")

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
