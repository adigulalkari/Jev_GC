# jev-gc

Real-time, OpenTelemetry-driven context garbage collection for LLM agents,
powered by [Jev](https://typesafe.ai) (TypeSafe AI's "System One" model).

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

[`examples/strands_dummy_agent/`](./examples/strands_dummy_agent/): a
Gemini-backed Strands agent with real tools (live weather, calculator,
docs search), 4 turns, through jev-gc end to end against the live Jev API:

```
spans_processed: 27   spans_kept_hot: 26   spans_kept_warm: 1
jev_calls: 1           jev_call_errors: 0
```

Most spans resolved free via the pre-filter, as designed — the ambiguous
bucket (and Jev's share of the work) grows with session length, not with
this 4-turn demo. See [`docs/architecture.md`](./docs/architecture.md).

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
pip install -e ".[dev,langgraph,strands]"
```

## Quickstart

```python
from opentelemetry.sdk.trace import TracerProvider
from jevgc import JevGC

provider = TracerProvider()
gc = JevGC.from_config("jevgc.yaml")            # reads JEV_API_KEY from env
gc.attach_to_tracer_provider(provider)

# ... agent runs, spans stream in automatically ...

prompt_context = gc.build_context(
    task="Reconcile invoice #4521 totals across ERP and warehouse",
    budget_tokens=8000,
)
```

No OTel yet? Use the manual escape hatch: `await gc.observe(span_record)`.
See [`docs/quickstart.md`](./docs/quickstart.md).

## Docs

- [`docs/quickstart.md`](./docs/quickstart.md)
- [`docs/architecture.md`](./docs/architecture.md)
- [`docs/jev_question_design.md`](./docs/jev_question_design.md)
- [`SPEC.md`](./SPEC.md)

## License

MIT
