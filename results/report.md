# jev-gc overhead benchmark — results

Every number below is derived from [`benchmark.json`](./benchmark.json), which
is the unmodified stdout of the run described here. Ratios, percentages and
bytes-per-entry are arithmetic on those fields and are labelled as such. No
figure that the run did not produce appears in this document.

## What was measured

`scripts/benchmark_overhead.py` drives a synthetic agent session through the
full jev-gc pipeline — prefilter → scorer → policy → render → store → archive —
at four span counts (10, 100, 500, 1000) and reports latency, memory, tier
distribution and token accounting at each point.

Jev itself is a `FakeJevClient` with a deterministic responder, so the timings
are jev-gc's own compute and nothing else. Each sweep point runs in its own
subprocess (`isolated_subprocess_per_count: true`) so one point's allocator
arena cannot contaminate the next point's RSS reading.

The whole sweep was run **five times** (`repeats: 5`). Every latency and memory
figure below is the **median of those five sweeps**, with the observed
**min–max range** shown alongside. Deterministic fields — tier counts, token
counts, archive sizes — are reported as single values, which the JSON confirms
is legitimate (see [Determinism check](#determinism-check)).

### Environment

Taken verbatim from `config.platform` in the JSON:

| | |
|---|---|
| Machine | arm64 |
| System | Darwin 25.5.0 |
| Python | 3.10.7 |
| Interpreter | `.venv/bin/python` |

The JSON records no commit hash, CPU model, core count or RAM size, so none is
claimed here.

### Reproduce

```
.venv/bin/python scripts/benchmark_overhead.py --repeat 5 --json > results/benchmark.json
```

Run configuration, as recorded in the `config` object of the JSON:
`seed=1337`, `spans_per_turn=10`, `batch_max_wait_ms=0`, `budget_tokens=8000`,
`jev_client=FakeJevClient`, `isolated_subprocess_per_count=true`, `repeats=5`.
Span counts are the script's defaults (10, 100, 500, 1000).

Span mix (exact proportions, dealt out then shuffled, not sampled):
relevant 42%, deadend 24%, pointer 14%, erroring 8%, trivial 8%, pinned 4%.

The generator is seeded, so a re-run at the same seed on the same commit
reproduces the tier counts, archive counts and token counts exactly. The timing
and RSS columns will differ run to run; that is what the five repeats are for.

## Latency

`observe total` is the wall time to feed all spans through `observe()`.
`per span` is that total divided by the span count. `build_context` and
`search_cold` are each the mean of 5 in-process repeats taken after the session
is fully loaded; the median and range below are then taken across the five
sweeps.

| spans | turns | observe total (ms) median [min–max] | per span (µs) median [min–max] | build_context (ms) median [min–max] | search_cold (ms) median [min–max] | Jev batches | Jev calls |
|------:|------:|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 0.842 [0.819–0.902] | 84.24 [81.88–90.22] | 0.012 [0.012–0.013] | 0.014 [0.014–0.014] | 1 | 10 |
| 100 | 10 | 9.023 [8.802–9.479] | 90.23 [88.02–94.79] | 0.081 [0.081–0.099] | 0.134 [0.133–0.150] | 10 | 88 |
| 500 | 50 | 45.993 [45.295–48.659] | 91.99 [90.59–97.32] | 0.390 [0.373–0.403] | 0.754 [0.744–0.765] | 50 | 440 |
| 1000 | 100 | 96.100 [94.308–107.367] | 96.10 [94.31–107.37] | 0.838 [0.733–0.940] | 1.570 [1.543–1.719] | 100 | 880 |

`Jev batches` and `Jev calls` were identical in all five sweeps at every point.
`Jev calls` is below the span count from 100 spans upward because the `trivial`
(8%) and `pinned` (4%) kinds short-circuit in the prefilter and never reach the
scorer. At 10 spans the exact-proportion dealer allocates zero of both kinds, so
all 10 spans are scored.

### Does per-span cost stay flat as span count grows?

**No — it rises, slightly but genuinely.** The median per-span `observe()` cost
goes from **84.24 µs at 10 spans to 96.10 µs at 1000 spans: a 1.14x rise while
span count grew 100x.**

The important question is whether 1.14x is a real trend or scheduling jitter.
The five repeats answer it directly:

- The observed range at 10 spans is **[81.88, 90.22] µs**.
- The observed range at 1000 spans is **[94.31, 107.37] µs**.
- **These ranges do not overlap.** The slowest of five sweeps at 10 spans
  (90.22 µs) is still faster than the fastest of five sweeps at 1000 spans
  (94.31 µs).

So the endpoint difference is not explained by run-to-run noise on this machine:
the per-span cost really is higher at 1000 spans than at 10. Calling it "flat"
would be wrong. Calling it a scaling problem would also be wrong — 1.14x for a
100x input growth is a very small, strongly sub-linear increase, and in absolute
terms it is 12 µs per span.

Between adjacent points the picture is less clear-cut, and this is where the
noise caveat bites. Every adjacent pair of ranges **does** overlap
(10 vs 100, 100 vs 500, 500 vs 1000), and 100 vs 1000 overlaps as well. Only the
10-vs-500 and 10-vs-1000 comparisons separate cleanly. Read the step table as
direction, not as three independently significant measurements:

| step | per span (µs), median | step ratio |
|---|---|---:|
| 10 → 100 spans | 84.24 → 90.23 | 1.07x |
| 100 → 500 spans | 90.23 → 91.99 | 1.02x |
| 500 → 1000 spans | 91.99 → 96.10 | 1.05x |

No single step dominates; the rise is spread thinly across the sweep, which is
consistent with a small real cost that grows with session size rather than a
cliff at one particular size.

### build_context() and search_cold()

Both are **per-turn** calls, not per-span calls. An agent calls `build_context()`
once when assembling a prompt and `search_cold()` only when it wants to recover
evicted material — so these costs are paid once per turn, while the `observe()`
cost above is paid once per span. Comparing their growth ratios to the per-span
figure directly would be a category error.

| | 10 spans | 1000 spans | growth | span growth |
|---|---:|---:|---:|---:|
| `build_context()` median | 0.012 ms | 0.838 ms | 69.81x | 100x |
| `search_cold()` median | 0.014 ms | 1.570 ms | 109.74x | 100x |

`build_context()` grows **sub-linearly** (69.81x for 100x the spans). Normalised
per span in the session it is essentially constant — 1.20, 0.81, 0.78, 0.84 µs
per span across the sweep.

`search_cold()` grows **109.74x for 100x the spans**, marginally faster than the
input. It is the one figure in this sweep that scales slightly worse than span
count. In absolute terms it is 1.570 ms at 1000 spans, once per search, which is
not a practical concern at this size; the scaling is what to watch if sessions
get much longer. `search_cold()` returned 5 hits (the requested limit) at 100,
500 and 1000 spans, and 2 hits at 10 spans, where only 2 spans reached COLD —
so the 10-span timing is doing strictly less work than the rest of the column.

## Memory

`peak RSS` is `resource.getrusage(RUSAGE_SELF).ru_maxrss` — a high-water mark,
so it is an upper bound on what is retained, not an exact retained figure.
`RSS delta` is measured against a post-warmup baseline in the same process.
`archive entries`, `archive retained` and `released` come from `Archive.stats()`,
confirmed as the source at every sweep point in all five runs; no fallback was
used. Archive figures were byte-identical across repeats, so they carry no range.

| spans | peak RSS (MB) median [min–max] | RSS delta (MB) median [min–max] | archive entries | archive retained (KB) | bytes/entry | released |
|------:|---:|---:|---:|---:|---:|---:|
| 10 | 42.75 [42.58–42.91] | 0.00 [0.00–0.02] | 3 | 1.24 | 421.7 | 0 |
| 100 | 43.59 [43.39–43.73] | 0.78 [0.77–0.80] | 54 | 21.32 | 404.3 | 0 |
| 500 | 47.48 [47.33–48.05] | 4.56 [4.48–4.67] | 270 | 107.92 | 409.3 | 0 |
| 1000 | 52.20 [50.45–52.59] | 9.44 [9.33–9.70] | 540 | 208.68 | 395.7 | 0 |

Archive entry count is exactly WARM + COLD at every point (220 + 320 = 540 at
1000 spans), and retained bytes per entry stays in a 395.7–421.7 byte band, so
archived text grows linearly with the evicted-span count at a stable per-entry
cost.

**Archived text as a share of the RSS delta** (derived: retained bytes ÷ delta):

| spans | archive retained | RSS delta | archived share of delta |
|------:|---:|---:|---:|
| 10 | 1.24 KB | 0.00 MB | not computable (delta is 0.00) |
| 100 | 21.32 KB | 0.78 MB | 2.66% |
| 500 | 107.92 KB | 4.56 MB | 2.31% |
| 1000 | 208.68 KB | 9.44 MB | 2.16% |

At 1000 spans the archived full text is **2.16%** of the 9.44 MB delta. The
other ~98% is the live per-span object graph (one `ContextItem` /
`GCDecision` / `EvictionEvent` set per span), which this benchmark measures only
in aggregate and does not break down. If you are worried about jev-gc's memory
cost, the archive is not the thing to worry about at these sizes; the retained
object graph is.

Two things worth flagging rather than smoothing over:

- **`archive_released_count` is 0 at every sweep point, in all five runs.** The
  archive's content-release path — the memory ceiling added in `ed3982b` — never
  fired. 208.68 KB of retained text is far below any plausible pressure
  threshold, so this is consistent with the feature working as designed, but it
  means **this benchmark does not exercise release at all**. Nothing here
  validates that the bound holds under pressure.
- **RSS delta at 10 spans is 0.00 MB at the median** (range 0.00–0.02 MB). That
  is `ru_maxrss` being a high-water mark: the warmup session's peak was never
  exceeded by a 10-span session. It is not evidence that a 10-span session costs
  nothing, and it is why the archived-share column has no entry on that row.

## Tiers and token accounting

Token counts are cumulative `GCStats` over the whole session, not a single
prompt. `full content` is the baseline of rendering every span in full;
`rendered` is what jev-gc actually emitted. All values in this table were
identical across all five repeats.

| spans | HOT | WARM | COLD | rendered | full content | saved | saved % | context chars |
|------:|----:|-----:|-----:|---------:|-------------:|------:|--------:|--------------:|
| 10 | 7 | 1 | 2 | 699 | 969 | 270 | 27.86% | 2,715 |
| 100 | 46 | 22 | 32 | 4,637 | 8,837 | 4,200 | 47.53% | 16,989 |
| 500 | 230 | 110 | 160 | 22,968 | 44,288 | 21,320 | 48.14% | 32,466 |
| 1000 | 460 | 220 | 320 | 46,444 | 87,262 | 40,818 | 46.78% | 32,481 |

`saved %` is `tokens_saved_estimate / tokens_full_content`, computed from the
table's own columns. Tier counts sum to the span count at every point.

**`tokens_saved_estimate` is now exact.** At every sweep point, in every one of
the five runs, `tokens_saved_estimate` equals `tokens_full_content −
tokens_rendered` precisely: 969 − 699 = 270, 8,837 − 4,637 = 4,200,
44,288 − 22,968 = 21,320, 87,262 − 46,444 = 40,818. An earlier version of this
report documented a discrepancy of +16 / +80 / +160 tokens caused by
`src/jevgc/telemetry.py` accumulating `max(0, full − rendered)` per item, which
clamped away the cases where a summary rendered larger than the content it
replaced and so overstated the headline saving. That clamp is gone — the counter
is now the plain net `full_token_count - token_count` and can go negative — and
this JSON confirms the stat and the net figure agree. **That finding is resolved
and is not an outstanding issue.**

The 10-span point saves a much smaller share (27.86%) than the rest. With one
turn and 10 spans the exact-proportion dealer produces only 7 relevant, 1
pointer and 2 deadend spans — no erroring, trivial or pinned spans at all — so
only 2 spans reach COLD. That row should not be read as a small-session result;
it is a small-sample result.

`context chars` plateaus between 500 and 1000 spans (32,466 → 32,481). That is
the 8,000-token `build_context` budget binding: past roughly 500 spans the
assembled context is budget-capped, not content-capped. Cumulative `rendered`
keeps growing because it counts every turn's render, not the final prompt.

## Determinism check

`summarize()` marks any invariant field that differed between repeats by adding
a `<field>_NONDETERMINISTIC: true` key to that span count's summary entry. The
checked fields are `tier_hot`, `tier_warm`, `tier_cold`, `jev_calls`,
`tokens_rendered`, `tokens_full_content`, `tokens_saved_estimate`,
`archive_entry_count` and `archive_retained_bytes`.

**No `_NONDETERMINISTIC` key appears anywhere in the JSON.** Every one of those
fields is a single scalar rather than a list at all four span counts. The tier
counts, the token figures and the archive entry counts and retained byte totals
were therefore **identical across all five runs**. A separate check of the raw
`runs` array confirms the same for `jev_batches`, `build_context_chars`,
`search_cold_hits`, `archive_released_count` and `archive_stats_source`.

The seeded generator and the fixed fake responder do what they claim: at a given
seed, the decisions this library makes are reproducible. Only the timing and RSS
columns vary.

## What this does not measure

This section matters more than the tables. Read it before quoting any number
above.

- **Nothing here measures Jev's judgment quality.** Every Jev answer comes from
  `FakeJevClient` with a fixed, deterministic responder keyed on the span kind
  the generator already knows. The fake returns the "right" answer by
  construction. Whether real Jev would tier these spans the same way is
  completely untested here, and the tier table above is therefore a measurement
  of the fixture's routing, not of Jev's accuracy.
- **Nothing here measures real network latency.** The fake client answers
  in-process with no I/O. Real Jev round-trips (documented in `SPEC.md` §1 as
  roughly 70–500 ms per *batch*) are absent from every latency column. At 1000
  spans the run issued 100 batches; a real deployment would add that batch
  latency on top of the 96.10 ms of jev-gc compute measured here. These are
  jev-gc's own pipeline costs, not end-to-end agent latency.
- **The span mix is synthetic and chosen to exercise every decision path.** The
  six kinds and their proportions were picked so that every branch carries real
  traffic — prefilter KEEP and DROP short-circuits, scorer Track A (relevance)
  and Track B (error treatment), policy keep and evict. A real workload has a
  different mix. **The token-savings percentage (46.78–48.14% at 100 spans and
  above) is a property of this fixture, not a forecast for any real agent.**
  Change the share of bulky dead-end spans and the percentage moves with it.
- **The config is not the default config, and is chosen to be pessimistic.**
  `keep_last_n_turns=0`, plus observing each turn's spans at the start of the
  *next* turn, is specifically arranged so spans do not short-circuit to KEEP in
  the prefilter. `batch_max_wait_ms=0` excludes the scorer's deliberate 200 ms
  batch window from the timings. The first choice makes the latency numbers an
  **upper bound** — a default-config session takes the cheap prefilter path more
  often and costs less. The second removes a latency the real system does incur
  by policy; it is excluded as a policy decision rather than pipeline overhead,
  but it is not zero in production. `emit_self_metrics=False` likewise removes
  OpenTelemetry's own per-span cost.
- **Five repeats on one machine is not a distribution across hardware.** The
  repeats establish run-to-run spread on a single arm64 Darwin 25.5.0 machine
  running CPython 3.10.7, which was not otherwise quiesced. They say nothing
  about x86-64, other Python versions, containerised or shared CPU, or a
  machine under concurrent load. The min–max ranges are the spread of five
  observations, not a confidence interval, and adjacent span counts' ranges
  overlap (see the latency section) — only the 10-vs-500 and 10-vs-1000
  comparisons separate cleanly.
- **The archive's memory ceiling is never exercised.** `archive_released_count`
  is 0 at every span count in every run. The bound added in `ed3982b` was never
  reached, so this benchmark provides no evidence at all about its behaviour
  under memory pressure — only that it does not fire when it should not.
- **The sweep stops at 1000 spans.** The per-span curve rises 1.14x across two
  decades of span count. Nothing here says what happens at 10,000, and the shape
  of the curve between 1000 and 10,000 is not measured. Any extrapolation beyond
  1000 spans is extrapolation, not measurement.
- **`tokens_saved_estimate` is not a cost estimate under a prefix-based prompt
  cache.** This benchmark counts tokens sent per render; it does not model a
  provider's prompt cache at all. Evicting or rewriting an early span changes
  every later byte's position in the prompt, so a cache keyed on a shared
  prefix can miss on the unchanged tail too — turning what looks like a token
  saving into more *uncached* (i.e. full-price) tokens than a stable,
  never-evicted prompt would have sent, even though the GC'd prompt is
  shorter. Neither this script nor `results/benchmark.json` records cached
  vs. uncached input tokens, so the percentages above say nothing about
  dollar cost under caching and should not be read as if they did.
