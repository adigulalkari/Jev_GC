"""Overhead benchmark: what does jev-gc cost on a run with hundreds of spans?

WHY THIS EXISTS
---------------
The question this answers, asked publicly of the maintainer, is narrow and
fair: *does the deterministic rule-evaluation cost stay flat as span count
grows, or does it creep up?* -- and, now that eviction retains an immutable
archived snapshot per evicted span (`archive.py`), *how much memory does a
long run actually hold?*

Someone deciding whether to put this library inside their agent loop reads
these numbers. So the script is built to be unflattering if the truth is
unflattering: it prints the per-span cost at every sweep point, states the
growth ratio, and labels a growing curve as growing.

WHAT IS AND IS NOT MEASURED
---------------------------
* Jev itself is a `FakeJevClient` -- no network, no API key. So every
  number here is **jev-gc's own compute**: prefilter -> scorer bookkeeping
  -> policy -> render -> store -> archive. Real Jev latency (~70-500ms per
  *batch*, per SPEC.md §1) is not included and is not jev-gc's overhead to
  own.
* `--batch-wait-ms` defaults to 0 so the scorer's deliberate batch window
  (`scorer.batch_max_wait_ms`, 200ms by default) is not counted as
  overhead. That window is a latency *policy*, not a cost; counting it
  would drown the signal we are actually after. The observed batch count is
  printed so the reader can see how much batching actually happened.
* Memory is `resource.getrusage` **peak** RSS, which is an upper bound on
  what is retained, not an exact retained figure. The exact retained figure
  for the thing that actually grows without bound -- archived content -- is
  reported separately as `archive retained` from `Archive.stats()`.
* Each sweep point runs in its own subprocess, so one span count's
  allocator arena cannot contaminate the next one's RSS delta. Pass
  `--in-process` to skip that; the RSS column is then not trustworthy and
  the script says so.

WHY THE CONFIG IS NOT THE DEFAULT CONFIG
----------------------------------------
With stock settings (`keep_last_n_turns=3`) a span observed on the turn it
was emitted matches prefilter rule 4 and short-circuits to KEEP without
ever reaching the scorer or the policy. Benchmarking that would measure
almost nothing. So this script models the other, more expensive shape: the
host observes a turn's spans at the *start of the next turn*, with
`keep_last_n_turns=0`, so the ambiguous majority genuinely walks the
scorer/policy/archive path. **These numbers are therefore an upper bound**
-- a default-config session takes the cheap prefilter path more often.

Run:
    .venv/bin/python scripts/benchmark_overhead.py
    .venv/bin/python scripts/benchmark_overhead.py --counts 10,100,500 --json

Deterministic: all synthetic content is drawn from an explicitly seeded
`random.Random`, so a re-run with the same `--seed` reproduces the same
session (SPEC.md §2 principle 7).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gc as stdlib_gc
import importlib
import json
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from typing import Any

import jevgc.archive
import jevgc.gc
import jevgc.store
from jevgc.archive import Archive, ColdIndexEntry
from jevgc.config import JevGCConfig
from jevgc.gc import JevGC
from jevgc.jev_client.client import JevBatchRequest
from jevgc.jev_client.fakes import FakeJevClient
from jevgc.models import (
    ErrorTreatment,
    JevChoiceAnswer,
    JevQuestionResult,
    JevScoreAnswer,
    SpanRecord,
    SpanStatus,
    Tier,
    Treatment,
)

DEFAULT_COUNTS = (10, 100, 500, 1000)
DEFAULT_SEED = 1337
DEFAULT_SPANS_PER_TURN = 10
DEFAULT_BUDGET_TOKENS = 8000
WARMUP_SPANS = 25
LATENCY_REPEATS = 5

# Question ids the scorer uses on the wire. Mirrored as literals rather than
# imported, because `scorer._Q_*` are private to that module -- same thing
# tests/integration/test_pipeline.py does.
Q_RELEVANCE = "relevance"
Q_ERROR_TREATMENT = "error_treatment"

SEARCH_QUERY = "warehouse pallet manifest rotterdam depot"

# `Archive.stats()` may be landing concurrently with this script; poll for it
# before falling back to a cruder estimate.
ARCHIVE_STATS_POLL_ATTEMPTS = 3
ARCHIVE_STATS_POLL_DELAY_S = 2.0

# macOS reports ru_maxrss in bytes; Linux reports it in kilobytes.
_RU_MAXRSS_SCALE = 1 if sys.platform == "darwin" else 1024


@dataclasses.dataclass(frozen=True)
class CountResult:
    """One sweep point. Every field is something the script measured; nothing
    here is estimated unless its name says so."""

    spans: int
    turns: int
    observe_total_s: float
    observe_per_span_us: float
    build_context_ms: float
    build_context_chars: int
    search_cold_ms: float
    search_cold_hits: int
    peak_rss_mb: float
    peak_rss_delta_mb: float
    archive_entry_count: int
    archive_retained_bytes: int
    archive_released_count: int | None
    archive_stats_source: str
    tier_hot: int
    tier_warm: int
    tier_cold: int
    jev_batches: int
    jev_calls: int
    tokens_rendered: int
    tokens_full_content: int
    tokens_saved_estimate: int


# --------------------------------------------------------------------------
# Synthetic session
# --------------------------------------------------------------------------

# Kind -> share of the session. A benchmark that fed one uniform span shape
# would measure one hot path; this mix makes the prefilter's KEEP and DROP
# short-circuits, the scorer's Track A (relevance) and Track B (error
# treatment), and the policy's keep/evict branches all carry real traffic.
SPAN_MIX: tuple[tuple[str, float], ...] = (
    ("relevant", 0.42),  # scored high -> HOT, rendered in full
    ("deadend", 0.24),  # scored low + DROP -> COLD, archived
    ("pointer", 0.14),  # scored low + KEEP_POINTER_ONLY -> WARM, archived
    ("erroring", 0.08),  # prefilter KEEP, scorer Track B
    ("trivial", 0.08),  # prefilter DROP, Jev never called
    ("pinned", 0.04),  # prefilter KEEP, Jev never called
)


def _weighted_kinds(rng: random.Random, count: int) -> list[str]:
    """Deal out `count` span kinds in the proportions of `SPAN_MIX`, then
    shuffle, so the mix is exact at every sweep point rather than only
    approaching the target ratios at large N (a 10-span run sampled
    independently could easily contain zero error spans)."""
    kinds: list[str] = []
    for kind, share in SPAN_MIX:
        kinds.extend([kind] * int(count * share))
    while len(kinds) < count:
        kinds.append("relevant")
    kinds = kinds[:count]
    rng.shuffle(kinds)
    return kinds


def _make_span(kind: str, index: int, turn_index: int, rng: random.Random) -> SpanRecord:
    """Build one realistically-shaped span. Payload sizes vary because the
    render/token-estimate path is length-sensitive, and a fixed-size fixture
    would hide that."""
    span_id = f"{kind}-{index}"
    start_ns = index * 1_000_000
    end_ns = start_ns + rng.randrange(5, 800) * 1_000_000

    def build(
        name: str,
        *,
        status: SpanStatus = SpanStatus.OK,
        input_preview: str | None = None,
        output_preview: str | None = None,
        output_token_count: int | None = 0,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> SpanRecord:
        return SpanRecord(
            span_id=span_id,
            trace_id="bench-trace",
            name=name,
            status=status,
            start_time_unix_ns=start_ns,
            end_time_unix_ns=end_ns,
            turn_index=turn_index,
            input_preview=input_preview,
            output_preview=output_preview,
            output_token_count=output_token_count,
            error_type=error_type,
            error_message=error_message,
        )

    if kind == "erroring":
        frames = "\n".join(
            f'  File "/app/tools/carrier.py", line {rng.randrange(20, 900)}, in _request'
            for _ in range(rng.randrange(6, 20))
        )
        return build(
            "query_shipping_carrier_api",
            status=SpanStatus.ERROR,
            input_preview=f"GET /shipments/{rng.randrange(1000, 9999)}",
            error_type="CarrierAPITimeout",
            error_message=f"shipping_carrier_api timed out after 30s\nTraceback:\n{frames}",
        )

    if kind == "trivial":
        # OK status, aged out, no output -- prefilter rule 5's DROP branch.
        return build(
            "check_cache",
            input_preview=f"cache_probe key=inv-{rng.randrange(1000, 9999)}",
        )

    if kind == "pinned":
        return build(
            "user_constraint",
            input_preview="system",
            output_preview=(
                "Constraint: reconcile only invoice #4521; do not touch other invoices. "
                "Report any discrepancy over $50 explicitly."
            ),
            output_token_count=40,
        )

    if kind == "relevant":
        lines = "\n".join(
            f"  line {i}: {rng.randrange(1, 80)}x Unit-{chr(65 + i % 6)} "
            f"@ ${rng.randrange(10, 400)}.{rng.randrange(10, 99)}"
            for i in range(rng.randrange(4, 14))
        )
        return build(
            "query_erp_table",
            input_preview="SELECT * FROM erp.invoice_lines WHERE invoice_id = 4521",
            output_preview=f"ERP invoice #4521 total ${rng.randrange(1000, 9000)}.00\n{lines}",
            output_token_count=rng.randrange(120, 400),
        )

    if kind == "pointer":
        return build(
            "query_erp_table",
            input_preview="SELECT * FROM erp.invoice_lines WHERE invoice_id = 4521",
            output_preview=(
                "ERP invoice #4521 header (repeat lookup): vendor Acme Logistics, "
                f"terms NET30, issued 2026-0{rng.randrange(1, 9)}-1{rng.randrange(0, 9)}"
            ),
            output_token_count=rng.randrange(40, 120),
        )

    # "deadend": plausible, bulky, and genuinely unrelated to the task --
    # the shape jev-gc exists to evict. Carries the SEARCH_QUERY keywords so
    # search_cold() has real (not empty) work to do at every sweep point.
    skus = ", ".join(f"SKU-{rng.randrange(10000, 99999)}" for _ in range(rng.randrange(20, 70)))
    return build(
        "query_warehouse_shipments",
        input_preview="SELECT * FROM wms.pallet_manifest WHERE depot = 'rotterdam'",
        output_preview=f"warehouse pallet manifest, rotterdam depot: {skus}",
        output_token_count=rng.randrange(200, 700),
    )


def bench_responder(requests: list[JevBatchRequest]) -> list[JevQuestionResult]:
    """Deterministic fake Jev, routed by the span-kind prefix in `item_id`.

    Fixed answers per kind, not random ones: the point of the benchmark is
    that two runs at the same seed exercise the same policy branches, so any
    difference in the timings is the library's, not the fixture's.
    """
    scores = {"relevant": 0.88, "pointer": 0.18, "deadend": 0.06}
    treatments = {
        "relevant": Treatment.INCLUDE_FULL.value,
        "pointer": Treatment.KEEP_POINTER_ONLY.value,
        "deadend": Treatment.DROP.value,
    }

    out: list[JevQuestionResult] = []
    for req in requests:
        kind = req.item_id.split("-", 1)[0]
        if req.question_id == Q_ERROR_TREATMENT:
            out.append(
                JevQuestionResult(
                    item_id=req.item_id,
                    question_id=req.question_id,
                    choice=JevChoiceAnswer(
                        option=ErrorTreatment.KEEP_ERROR_SUMMARY_ONLY.value, probability=0.82
                    ),
                )
            )
        elif req.question_id == Q_RELEVANCE:
            out.append(
                JevQuestionResult(
                    item_id=req.item_id,
                    question_id=req.question_id,
                    score=JevScoreAnswer(score=scores.get(kind, 0.5), confidence=0.9),
                )
            )
        else:
            out.append(
                JevQuestionResult(
                    item_id=req.item_id,
                    question_id=req.question_id,
                    choice=JevChoiceAnswer(
                        option=treatments.get(kind, Treatment.INCLUDE_SUMMARY_ONLY.value),
                        probability=0.8,
                    ),
                )
            )
    return out


def bench_config(batch_wait_ms: int) -> JevGCConfig:
    """See the module docstring: `keep_last_n_turns=0` plus a one-turn
    observation lag is what keeps the ambiguous majority on the expensive
    path instead of short-circuiting in the prefilter.

    `emit_self_metrics=False` because OTel's no-op meter would otherwise add
    a per-span cost that belongs to OpenTelemetry, not to jev-gc's GC logic.
    """
    return JevGCConfig.from_dict(
        {
            "jev": {"api_key": "unused-fake-client"},
            "prefilter": {"keep_last_n_turns": 0, "drop_zero_output_after_turns": 0},
            "scorer": {"batch_max_size": 50, "batch_max_wait_ms": batch_wait_ms},
            "policy": {"relevance_keep_threshold": 0.35, "min_confidence_to_drop": 0.6},
            "telemetry": {"emit_self_metrics": False},
        }
    )


# --------------------------------------------------------------------------
# Memory probes
# --------------------------------------------------------------------------


def peak_rss_bytes() -> int:
    """Peak resident set size of this process. Stdlib only: `psutil` is not a
    declared dependency of jev-gc and a benchmark should not add one."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * _RU_MAXRSS_SCALE


def wait_for_archive_stats() -> bool:
    """Poll for `Archive.stats()`, which may be landing concurrently.

    Reloads the three modules that bind `Archive` by value, in dependency
    order, so a freshly-written method is actually picked up rather than
    masked by the copy imported at process start. A partially-written source
    file raises here; that just counts as "not ready yet".
    """
    for attempt in range(ARCHIVE_STATS_POLL_ATTEMPTS):
        if hasattr(jevgc.archive.Archive, "stats"):
            return True
        if attempt == ARCHIVE_STATS_POLL_ATTEMPTS - 1:
            break
        time.sleep(ARCHIVE_STATS_POLL_DELAY_S)
        try:
            for module in (jevgc.archive, jevgc.store, jevgc.gc):
                importlib.reload(module)
        except Exception:  # mid-write source file: count as "not ready", retry
            continue
    return hasattr(jevgc.archive.Archive, "stats")


def probe_archive(archive: Archive) -> tuple[int, int, int | None, str]:
    """(entry_count, retained_bytes, released_count, source).

    Prefers `Archive.stats()`. Falls back to a `sys.getsizeof` walk over
    `entries()` and *says so* in the returned source string, because the
    fallback measures the `str` objects' own footprint only -- it cannot see
    entries whose text was released under memory pressure, so it has no
    `released_count` to report and must return None rather than 0.
    """
    stats_fn = getattr(archive, "stats", None)
    if callable(stats_fn):
        stats = stats_fn()
        return (
            int(stats.entry_count),
            int(stats.retained_bytes),
            int(stats.released_count),
            "Archive.stats()",
        )

    entries = archive.entries()
    retained = sum(sys.getsizeof(entry.full_text) for entry in entries)
    return len(entries), int(retained), None, "sys.getsizeof fallback (Archive.stats() absent)"


# --------------------------------------------------------------------------
# One sweep point
# --------------------------------------------------------------------------


async def _drive(jgc: JevGC, spans: list[SpanRecord], spans_per_turn: int) -> None:
    """Feed `spans` through `observe()` turn by turn.

    Spans within a turn are gathered rather than awaited one at a time: the
    scorer batches by design (SPEC.md §2 principle 6), and awaiting each
    `observe` serially would force a batch of one per span and measure the
    batch timer instead of the pipeline.
    """
    for start in range(0, len(spans), spans_per_turn):
        chunk = spans[start : start + spans_per_turn]
        # Observe the previous turn's spans at the start of this turn, so
        # `turns_ago == 1` and the recency short-circuit does not fire.
        jgc.advance_turn(chunk[0].turn_index + 1)
        await asyncio.gather(*(jgc.observe(span) for span in chunk))


async def run_count(
    count: int,
    *,
    seed: int,
    spans_per_turn: int,
    batch_wait_ms: int,
    budget_tokens: int,
    archive_stats_available: bool,
) -> CountResult:
    """Measure one span count end to end, in this process."""
    rng = random.Random(seed)
    config = bench_config(batch_wait_ms)

    # Warm up interpreter/pydantic/asyncio paths with a fixed-size throwaway
    # session, identical at every sweep point, so first-call costs do not land
    # disproportionately on the smallest count and fake a "growing" curve.
    warmup_rng = random.Random(seed + 1)
    warmup_gc = jevgc.gc.JevGC(config, jev_client=FakeJevClient(responder=bench_responder))
    warmup_spans = [
        _make_span(kind, i, i // spans_per_turn, warmup_rng)
        for i, kind in enumerate(_weighted_kinds(warmup_rng, WARMUP_SPANS))
    ]
    await _drive(warmup_gc, warmup_spans, spans_per_turn)
    warmup_gc.build_context(task="warmup", budget_tokens=budget_tokens)
    del warmup_gc, warmup_spans
    stdlib_gc.collect()

    baseline_rss = peak_rss_bytes()

    kinds = _weighted_kinds(rng, count)
    spans = [_make_span(kind, i, i // spans_per_turn, rng) for i, kind in enumerate(kinds)]

    client = FakeJevClient(responder=bench_responder)
    jgc = jevgc.gc.JevGC(config, jev_client=client)
    for span in spans:
        if span.span_id.startswith("pinned-"):
            jgc.pin(span.span_id)

    observe_start = time.perf_counter()
    await _drive(jgc, spans, spans_per_turn)
    observe_total = time.perf_counter() - observe_start

    task = "Reconcile invoice #4521 totals across ERP and warehouse shipment records"

    build_times: list[float] = []
    context = ""
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        context = jgc.build_context(task=task, budget_tokens=budget_tokens)
        build_times.append(time.perf_counter() - start)

    search_times: list[float] = []
    hits: list[ColdIndexEntry] = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        hits = jgc.search_cold(SEARCH_QUERY, limit=5)
        search_times.append(time.perf_counter() - start)

    stdlib_gc.collect()
    final_rss = peak_rss_bytes()

    # Deliberate private-attribute access: `_store` is not part of the public
    # facade (SPEC.md §4.8 keeps that surface small), but the archive's memory
    # footprint is exactly what this benchmark exists to report. This script is
    # a measurement tool, NOT an example of how to use the library.
    store = jgc._store
    entry_count, retained_bytes, released_count, source = probe_archive(store.archive)
    if not archive_stats_available:
        source = "sys.getsizeof fallback (Archive.stats() absent)"

    stats = jgc.stats()
    await jgc.aclose()

    return CountResult(
        spans=count,
        turns=(count + spans_per_turn - 1) // spans_per_turn,
        observe_total_s=observe_total,
        observe_per_span_us=observe_total / count * 1e6,
        build_context_ms=sum(build_times) / len(build_times) * 1e3,
        build_context_chars=len(context),
        search_cold_ms=sum(search_times) / len(search_times) * 1e3,
        search_cold_hits=len(hits),
        peak_rss_mb=final_rss / 1024 / 1024,
        peak_rss_delta_mb=(final_rss - baseline_rss) / 1024 / 1024,
        archive_entry_count=entry_count,
        archive_retained_bytes=retained_bytes,
        archive_released_count=released_count,
        archive_stats_source=source,
        tier_hot=len(store.list_by_tier(Tier.HOT)),
        tier_warm=len(store.list_by_tier(Tier.WARM)),
        tier_cold=len(store.list_by_tier(Tier.COLD)),
        jev_batches=jgc._scorer.batches_sent,
        jev_calls=stats.jev_calls,
        tokens_rendered=stats.tokens_rendered,
        tokens_full_content=stats.tokens_full_content,
        tokens_saved_estimate=stats.tokens_saved_estimate,
    )


def run_count_in_subprocess(count: int, args: argparse.Namespace) -> CountResult:
    """Run one sweep point in a fresh interpreter.

    RSS never shrinks back to the OS when a Python object graph is freed, so
    measuring several span counts in one process would charge count N's
    memory column with count N-1's arena. A subprocess per point is the only
    way to make the memory column mean what it says.
    """
    proc = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker-count",
            str(count),
            "--seed",
            str(args.seed),
            "--spans-per-turn",
            str(args.spans_per_turn),
            "--batch-wait-ms",
            str(args.batch_wait_ms),
            "--budget-tokens",
            str(args.budget_tokens),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"worker for {count} spans failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    payload = [line for line in proc.stdout.splitlines() if line.strip()][-1]
    return CountResult(**json.loads(payload))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _ratio(last: float, first: float) -> float | None:
    return last / first if first > 0 else None


def _fmt_ratio(value: float | None) -> str:
    return f"{value:.2f}x" if value is not None else "n/a"


def _scaling_label(latency_ratio: float | None, span_ratio: float) -> str:
    """Describe growth relative to how much the input grew, not in absolute
    terms -- a 6x slowdown for 100x more spans is sub-linear, i.e. fine."""
    if latency_ratio is None:
        return "unmeasurable (baseline too small to time)"
    if latency_ratio < span_ratio * 0.35:
        return "sub-linear"
    if latency_ratio <= span_ratio * 1.35:
        return "roughly linear"
    return "SUPER-LINEAR"


def print_report(results: list[CountResult], args: argparse.Namespace) -> None:
    print("jev-gc overhead benchmark")
    print(
        f"  seed={args.seed}  spans/turn={args.spans_per_turn}  "
        f"batch_max_wait_ms={args.batch_wait_ms}  build_context budget={args.budget_tokens} tokens"
    )
    print("  Jev client: FakeJevClient (no network, no API key) -- timings are jev-gc compute only")
    print(f"  span mix: {', '.join(f'{k} {int(s * 100)}%' for k, s in SPAN_MIX)}")
    print(f"  archive memory source: {results[0].archive_stats_source}")
    if args.in_process:
        print("  WARNING: --in-process; peak-RSS columns are contaminated across sweep points")
    print()

    print("Latency")
    header = (
        f"{'spans':>6} {'turns':>6} {'observe total':>14} {'per span':>11} "
        f"{'build_context':>14} {'search_cold':>12} {'jev batches':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.spans:>6} {r.turns:>6} {r.observe_total_s * 1e3:>11.1f} ms "
            f"{r.observe_per_span_us:>8.1f} us {r.build_context_ms:>11.3f} ms "
            f"{r.search_cold_ms:>9.3f} ms {r.jev_batches:>12}"
        )
    print()

    print("Memory and tiers")
    header2 = (
        f"{'spans':>6} {'peak RSS':>10} {'RSS delta':>11} {'archive entries':>16} "
        f"{'archive retained':>17} {'released':>9} {'HOT':>6} {'WARM':>6} {'COLD':>6}"
    )
    print(header2)
    print("-" * len(header2))
    for r in results:
        released = "n/a" if r.archive_released_count is None else str(r.archive_released_count)
        print(
            f"{r.spans:>6} {r.peak_rss_mb:>7.1f} MB {r.peak_rss_delta_mb:>8.2f} MB "
            f"{r.archive_entry_count:>16} {r.archive_retained_bytes / 1024:>14.1f} KB "
            f"{released:>9} {r.tier_hot:>6} {r.tier_warm:>6} {r.tier_cold:>6}"
        )
    print()

    print("Tokens (GCStats, cumulative over the session)")
    header3 = f"{'spans':>6} {'rendered':>12} {'full content':>14} {'saved':>12} {'saved %':>9} {'context chars':>14}"
    print(header3)
    print("-" * len(header3))
    for r in results:
        pct = (
            r.tokens_saved_estimate / r.tokens_full_content * 100
            if r.tokens_full_content
            else 0.0
        )
        print(
            f"{r.spans:>6} {r.tokens_rendered:>12} {r.tokens_full_content:>14} "
            f"{r.tokens_saved_estimate:>12} {pct:>8.1f}% {r.build_context_chars:>14}"
        )
    print()

    print_verdict(results)


def print_verdict(results: list[CountResult]) -> None:
    first, last = results[0], results[-1]
    span_ratio = last.spans / first.spans

    per_span_ratio = _ratio(last.observe_per_span_us, first.observe_per_span_us)
    build_ratio = _ratio(last.build_context_ms, first.build_context_ms)
    search_ratio = _ratio(last.search_cold_ms, first.search_cold_ms)

    print("Verdict")
    print("-" * 78)
    print(
        f"Per-span observe cost: {first.observe_per_span_us:.1f} us at {first.spans} spans "
        f"-> {last.observe_per_span_us:.1f} us at {last.spans} spans "
        f"({_fmt_ratio(per_span_ratio)} while span count grew {span_ratio:.0f}x)."
    )
    if per_span_ratio is None:
        print("  Could not compare: the smallest sweep point was too fast to time reliably.")
    elif per_span_ratio <= 1.25:
        print("  -> FLAT. Per-span rule-evaluation cost does not creep up with session size.")
    elif per_span_ratio <= 2.0:
        print(
            "  -> MILDLY GROWING. Per-span cost rose measurably; not constant, but far from linear."
        )
    else:
        print(
            "  -> GROWING. Per-span cost is NOT flat -- it rises with the number of spans already "
            "in the session. Treat long runs as a known cost."
        )

    # The endpoint ratio alone can hide a curve that jumps once and then
    # settles (or settles and then jumps). Name the largest single step so a
    # reader can see *where* any movement happens, not just the net.
    steps = [
        (prev, cur, cur.observe_per_span_us / prev.observe_per_span_us)
        for prev, cur in zip(results, results[1:], strict=False)
        if prev.observe_per_span_us > 0
    ]
    if steps:
        prev, cur, worst = max(steps, key=lambda step: step[2])
        print(
            f"  Largest single step: {prev.spans} -> {cur.spans} spans, "
            f"{prev.observe_per_span_us:.1f} -> {cur.observe_per_span_us:.1f} us ({worst:.2f}x)."
        )

    print(
        f"build_context(): {first.build_context_ms:.3f} ms -> {last.build_context_ms:.3f} ms "
        f"({_fmt_ratio(build_ratio)} for {span_ratio:.0f}x spans) -- "
        f"{_scaling_label(build_ratio, span_ratio)}."
    )
    print(
        f"search_cold():   {first.search_cold_ms:.3f} ms -> {last.search_cold_ms:.3f} ms "
        f"({_fmt_ratio(search_ratio)} for {span_ratio:.0f}x spans) -- "
        f"{_scaling_label(search_ratio, span_ratio)}."
    )

    if last.archive_entry_count:
        per_entry = last.archive_retained_bytes / last.archive_entry_count
        print(
            f"Archive retention: {last.archive_entry_count} entries holding "
            f"{last.archive_retained_bytes / 1024:.1f} KB of full text at {last.spans} spans "
            f"({per_entry:.0f} bytes/entry, measured). Extrapolating that rate (NOT measured), "
            f"a 10,000-span session would hold roughly "
            f"{per_entry * last.archive_entry_count / last.spans * 10_000 / 1024 / 1024:.1f} MB."
        )
    print(
        f"Process peak RSS at {last.spans} spans: {last.peak_rss_mb:.1f} MB total, "
        f"{last.peak_rss_delta_mb:.2f} MB above the pre-session baseline."
    )
    delta_bytes = last.peak_rss_delta_mb * 1024 * 1024
    if delta_bytes > 0:
        share = last.archive_retained_bytes / delta_bytes * 100
        print(
            f"  Of that delta, archived full text is {share:.1f}%. The remainder is the live "
            "per-span object graph (ContextItem / GCDecision / EvictionEvent, one set per span), "
            "which this script measures only in aggregate and does not break down further."
        )
    if last.archive_released_count is None:
        print(
            "  NOTE: Archive.stats() was unavailable; archive retention above is a "
            "sys.getsizeof estimate over entries() and cannot report released entries."
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--counts",
        default=",".join(str(c) for c in DEFAULT_COUNTS),
        help="comma-separated span counts to sweep (default: 10,100,500,1000)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (default: 1337)")
    parser.add_argument("--spans-per-turn", type=int, default=DEFAULT_SPANS_PER_TURN)
    parser.add_argument(
        "--batch-wait-ms",
        type=int,
        default=0,
        help="scorer.batch_max_wait_ms; 0 keeps the deliberate batch window out of the timings",
    )
    parser.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS)
    parser.add_argument(
        "--json", action="store_true", help="emit results as JSON instead of the tables"
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="run every sweep point in this process (faster, but RSS columns become meaningless)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "run the whole sweep N times and report median/min/max per metric. "
            "A single sweep on a laptop cannot distinguish a real trend from "
            "scheduling noise, so anything published should use N>=5"
        ),
    )
    parser.add_argument("--worker-count", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


_SUMMARY_FIELDS = (
    "observe_total_s",
    "observe_per_span_us",
    "build_context_ms",
    "search_cold_ms",
    "peak_rss_mb",
    "peak_rss_delta_mb",
)

#: Fields that are a pure function of the seed and span count. If one of these
#: varies between repeats the benchmark is not deterministic and the whole run
#: is suspect, so they are checked rather than averaged.
_INVARIANT_FIELDS = (
    "tier_hot",
    "tier_warm",
    "tier_cold",
    "jev_calls",
    "tokens_rendered",
    "tokens_full_content",
    "tokens_saved_estimate",
    "archive_entry_count",
    "archive_retained_bytes",
)


def summarize(runs: list[list[CountResult]]) -> list[dict[str, Any]]:
    """Collapse repeated sweeps into median/min/max per span count.

    Median rather than mean: one scheduling hiccup on a laptop can move a mean
    enough to invent a trend that is not there, which is exactly the error this
    flag exists to prevent.
    """
    summary: list[dict[str, Any]] = []
    for index, first in enumerate(runs[0]):
        points = [run[index] for run in runs]
        entry: dict[str, Any] = {"spans": first.spans, "turns": first.turns, "repeats": len(runs)}
        for field in _SUMMARY_FIELDS:
            values = [float(getattr(point, field)) for point in points]
            entry[field] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        for field in _INVARIANT_FIELDS:
            values_seen = {getattr(point, field) for point in points}
            entry[field] = values_seen.pop() if len(values_seen) == 1 else sorted(values_seen)
            if entry[field] != getattr(first, field):
                entry[f"{field}_NONDETERMINISTIC"] = True
        summary.append(entry)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    archive_stats_available = wait_for_archive_stats()

    if args.worker_count is not None:
        result = asyncio.run(
            run_count(
                args.worker_count,
                seed=args.seed,
                spans_per_turn=args.spans_per_turn,
                batch_wait_ms=args.batch_wait_ms,
                budget_tokens=args.budget_tokens,
                archive_stats_available=archive_stats_available,
            )
        )
        print(json.dumps(dataclasses.asdict(result)))
        return

    counts = [int(part) for part in args.counts.split(",") if part.strip()]
    if not counts:
        raise SystemExit("--counts must name at least one span count")

    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    runs: list[list[CountResult]] = []
    for _ in range(args.repeat):
        results: list[CountResult] = []
        for count in sorted(counts):
            if args.in_process:
                results.append(
                    asyncio.run(
                        run_count(
                            count,
                            seed=args.seed,
                            spans_per_turn=args.spans_per_turn,
                            batch_wait_ms=args.batch_wait_ms,
                            budget_tokens=args.budget_tokens,
                            archive_stats_available=archive_stats_available,
                        )
                    )
                )
            else:
                results.append(run_count_in_subprocess(count, args))
        runs.append(results)

    config = {
        "seed": args.seed,
        "spans_per_turn": args.spans_per_turn,
        "batch_max_wait_ms": args.batch_wait_ms,
        "budget_tokens": args.budget_tokens,
        "jev_client": "FakeJevClient",
        "isolated_subprocess_per_count": not args.in_process,
        "span_mix": dict(SPAN_MIX),
        "repeats": args.repeat,
        "platform": {
            "machine": platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
    }

    if args.json:
        payload: dict[str, Any] = {"config": config}
        if args.repeat == 1:
            payload["results"] = [dataclasses.asdict(r) for r in runs[0]]
        else:
            payload["summary"] = summarize(runs)
            payload["runs"] = [[dataclasses.asdict(r) for r in run] for run in runs]
        print(json.dumps(payload, indent=2))
        return

    print_report(runs[-1], args)
    if args.repeat > 1:
        print_repeat_summary(summarize(runs))


def print_repeat_summary(summary: list[dict[str, Any]]) -> None:
    """Print the spread across repeats, because the single-run tables above
    cannot show whether a difference between two span counts is a trend or
    noise -- and on a laptop it is usually noise."""
    repeats = summary[0]["repeats"] if summary else 0
    print(f"\nAcross {repeats} repeats -- observe cost per span, microseconds\n")
    print(f"{'spans':>7}{'median':>10}{'min':>10}{'max':>10}{'spread':>10}")
    for entry in summary:
        stat = entry["observe_per_span_us"]
        spread = stat["max"] - stat["min"]
        print(
            f"{entry['spans']:>7}{stat['median']:>10.1f}"
            f"{stat['min']:>10.1f}{stat['max']:>10.1f}{spread:>10.1f}"
        )
    if len(summary) >= 2:
        first = summary[0]["observe_per_span_us"]["median"]
        last = summary[-1]["observe_per_span_us"]["median"]
        span_growth = summary[-1]["spans"] / summary[0]["spans"]
        print(
            f"\nMedian per-span cost grew {last / first:.2f}x while span count "
            f"grew {span_growth:.0f}x."
        )


if __name__ == "__main__":
    main()
