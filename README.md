# jev-gc

Real-time, OpenTelemetry-driven context garbage collection for LLM agents,
powered by [Jev](https://typesafe.ai) (TypeSafe AI's "System One" model).

**[Documentation and measured overhead →](https://adigulalkari.github.io/Jev_GC/)**

## The problem

Long agent sessions accumulate tool results, retrieved chunks, and failed
calls faster than any prompt window can hold. All of it gets re-sent to
the LLM on every turn whether or not it's still relevant — wasting tokens
and burying what actually matters under stale noise ("context rot").

## What jev-gc does

Hooks into the OTel spans your agent already emits and decides, span by
span, in real time: keep verbatim, compress to a pointer, or archive.
Nothing is ever hard-deleted. Cheap deterministic rules (pins, errors,
recency, staleness) resolve most spans for free — only the genuinely
ambiguous ones go to Jev for a relevance/treatment judgment.

```
Agent tool call → OTel span emitted → jev-gc classifies → next prompt
                                            │
                           deterministic rules resolve most spans free;
                           only the ambiguous ones reach Jev
```

## Why a keyword rule isn't enough

Real output from [`scripts/relevance_demo.py`](./scripts/relevance_demo.py)
against the live Jev API. Task: *"Reconcile invoice #4521 totals across
the ERP system and warehouse shipment records."* Two tool results, same
tool, same shape — one caused the mismatch, one is unrelated noise:

| span | keyword-overlap score | Jev relevance score |
|---|---:|---:|
| shipment delay for *this* order | 0.11 | **0.75** |
| shipment delay for an *unrelated* order | 0.11 | **0.45** |
| line-item breakdown from 3 turns ago | 0.33 | **0.97** |

A keyword/BM25 rule gives the relevant and irrelevant spans the **same
score** — it has no way to know one belongs to invoice #4521 and the
other doesn't; that's a fact about meaning, not vocabulary. Jev separates
them by 0.30. Reproduce it: `export JEV_API_KEY=... && python scripts/relevance_demo.py`

## Real run

[`examples/codebase_triage_agent/`](./examples/codebase_triage_agent/): a
bug-triage agent that investigates real bug reports against this actual
repo's own source, tests, and git history — real `git grep`, real file
reads, real `pytest` runs, no mocked data — through jev-gc end to end
against the live Jev API. See that example's README for the real
`tokens_saved_estimate` numbers from a full run.

## Design

1. **Code calculates, Jev judges.** Thresholds/age/status/pins are plain
   code. Jev only sees what's left.
2. **Fail open.** Jev down, slow, or low-confidence → keep at current tier.
3. **Never hard-delete.** Evicted just means out of the *next* prompt.
4. **Errors are never dropped, only compressed.** Only payload size is
   negotiable via Jev — presence never is.

`JevClient` is a `Protocol`, not a hard dependency threaded through the
codebase — swap in another scorer behind the same interface if you want.
Full rationale in [`SPEC.md`](./SPEC.md).

## Install

```bash
pip install jev-gc
```

Optional extras: `pip install "jev-gc[langgraph]"` or `"jev-gc[strands]"` for the
respective integration hooks.

Contributing or running the examples/tests from a clone instead:

```bash
git clone https://github.com/adigulalkari/Jev_GC
cd Jev_GC
pip install -e ".[dev,langgraph,strands]"
```

## Quickstart

No config file needed -- `JevGCConfig` only requires the API key, everything
else defaults:

```python
import os
from opentelemetry.sdk.trace import TracerProvider
from jevgc import JevGC
from jevgc.config import JevGCConfig

provider = TracerProvider()
config = JevGCConfig.from_dict({"jev": {"api_key": os.environ["JEV_API_KEY"]}})
gc = JevGC(config)
gc.attach_to_tracer_provider(provider)

# ... agent runs, spans stream in automatically ...

prompt_context = gc.build_context(
    task="Reconcile invoice #4521 totals across ERP and warehouse",
    budget_tokens=8000,
)
```

Prefer a YAML file instead? `JevGC.from_config("jevgc.yaml")` loads one (see
[`config/jevgc.example.yaml`](./config/jevgc.example.yaml) in this repo --
note that file ships in the git checkout, not in the PyPI package, so copy
its contents rather than its path if you only `pip install`ed jev-gc).

No OTel yet? Use the manual escape hatch: `await gc.observe(span_record)`.
See [`docs/quickstart.md`](./docs/quickstart.md).

## Getting evicted context back

Eviction would be a one-way door if the only way to ask for a span were a
`span_id` the agent can no longer see. So every span leaving HOT is archived
as an immutable snapshot, and a keyword-only index of what's evicted stays
cheap enough to show the agent every turn:

```python
gc.cold_index()                      # keywords only, never content
gc.search_cold("rotterdam manifest") # -> [ColdIndexEntry(span_id=..., tier=COLD)]
gc.rehydrate(span_id)                # full content back in HOT
```

`rehydrate` resolves to the snapshot taken when the span was observed, not to
a live re-read of the original source — what comes back is what was actually
evicted, so an audit of that decision reads the same evidence the decision saw.

Retaining that content costs memory, so the archive runs under a budget
(`archive.max_content_bytes`, default 8 MiB). Past it, the least-recently-used
snapshots release their *content* and keep their *keywords*: the span stays in
the cold index, still answers a search, and still counts in a regret pass — it
just can't be restored verbatim anymore, and `rehydrate` degrades to a tier
promotion with a logged warning. Keywords cost ~100 bytes against content that
runs to kilobytes, so discoverability outlives restorability by a wide margin.
`gc.archive_stats()` reports `retained_bytes` and `released_count`; a
`released_count` above zero means you've hit the ceiling.

## Measuring false eviction

The failure mode that matters isn't a bad summary, it's a span that looked
irrelevant at turn 3 and mattered at turn 11 — silently, with no error. Every
tier transition is logged, and a shadow replay diffs a no-eviction baseline run
against the GC'd run:

```python
findings = gc.analyze_regret(
    baseline_output=full_context_run_answer,
    evicted_output=gc_run_answer,
)
```

Each `RegretFinding` names an evicted span whose distinctive terms surfaced
only in the baseline answer. It's keyword-overlap evidence, not proof of
causation — a ranked list of evictions worth inspecting and a dial for tuning
`relevance_keep_threshold`, not a regression gate.

Note what this shadow replay does *not* compare: prompt-cache usage between
the baseline and GC'd run. Under a prefix-based cache, evicting or rewriting
an early span changes every later byte's position in the prompt, so the
unchanged tail can miss the cache too — a shorter, GC'd prompt is not
automatically a cheaper one if it keeps breaking cache reuse that a stable,
never-evicted prompt would have kept hitting. `analyze_regret` and the
benchmarks below report token counts, not cached-vs-uncached input tokens, so
neither currently answers "does this save money," only "does this save
tokens" and "did an eviction visibly cost an answer."

## Docs

- **[adigulalkari.github.io/Jev_GC](https://adigulalkari.github.io/Jev_GC/)** —
  install, configuration, API reference, the measured overhead, and an
  explicit account of what those numbers do not prove
- [`docs/quickstart.md`](./docs/quickstart.md)
- [`docs/architecture.md`](./docs/architecture.md)
- [`docs/jev_question_design.md`](./docs/jev_question_design.md)
- [`SPEC.md`](./SPEC.md)

## License

MIT
