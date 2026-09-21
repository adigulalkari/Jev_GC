# jev-gc

Real-time, OpenTelemetry-driven context garbage collection for LLM agents,
powered by [Jev](https://typesafe.ai) (TypeSafe AI's "System One" decision model).

**Status:** design complete, implementation in progress. See [`SPEC.md`](./SPEC.md)
for the full architecture and build plan.

## What it does

Long-running agent sessions accumulate context faster than any single prompt
window can hold: tool results, retrieved chunks, failed calls, dead-end
investigations. jev-gc hooks into the OpenTelemetry spans your agent already
emits, classifies each one's relevance to the current task using Jev
(sub-second, sub-cent per call), and decides in real time what stays in the
next prompt verbatim, what gets compressed to a pointer, and what gets
archived — without ever hard-deleting anything or requiring a separate LLM
summarization pass.

```
Agent tool call → OTel span emitted → jev-gc classifies → next prompt
                                            │
                                for the ambiguous cases only;
                                cheap deterministic rules
                                (errors, recency, pins) handle
                                the rest without calling Jev at all
```

## Install

```bash
pip install jev-gc
# or, with the LangGraph integration:
pip install "jev-gc[langgraph]"
```

## Quickstart

```python
from jevgc import JevGC

gc = JevGC.from_config("jevgc.yaml")           # reads JEV_API_KEY from env
gc.attach_to_tracer_provider(tracer_provider)   # start observing spans

# ... agent runs normally, spans stream in automatically ...

prompt_context = gc.build_context(
    task="Reconcile invoice #4521 totals across ERP and warehouse",
    budget_tokens=8000,
)
```

See [`examples/strands_dummy_agent/`](./examples/strands_dummy_agent/) for a
complete, runnable example agent (built with [Strands Agents](https://strandsagents.com))
wired up to jev-gc end to end.

## Documentation

- [`SPEC.md`](./SPEC.md) — full architecture, module contracts, coding
  standards, and the build plan this repo is implemented against.
- [`docs/architecture.md`](./docs/architecture.md) — deeper dive once
  implementation lands.

## License

MIT
