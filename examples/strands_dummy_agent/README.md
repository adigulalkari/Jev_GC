# Strands dummy agent example

A small, runnable [Strands Agents](https://strandsagents.com) agent, backed
by Gemini (`gemini-2.0-flash`), with three tools that do real work:

| Tool | What it actually does |
|---|---|
| `get_weather(city)` | Live lookup via the free [Open-Meteo](https://open-meteo.com) API (geocoding + current temperature). No API key needed. |
| `calculate(expression)` | Safe AST-based arithmetic evaluator (no `eval`). |
| `search_docs(query)` | Greps this repo's `README.md` and `docs/*.md` for a term. |

The agent itself isn't the point -- it exists purely to emit realistic
OpenTelemetry GenAI spans for jev-gc to consume (SPEC.md §9). The scripted
prompts deliberately include one irrelevant aside (a calculation unrelated
to the running thread) and one tool call that errors (an invalid city
name), so there's real material for jev-gc's relevance scoring and
error-handling paths.

## Run it

```bash
pip install -e ".[examples]"
export JEV_API_KEY=...       # from the TypeSafe dashboard; omit to run against FakeJevClient
export GEMINI_API_KEY=...    # required -- this example needs a real LLM
python examples/strands_dummy_agent/dummy_agent.py
```

It runs four short turns, then prints:

- the assembled, budget-fit context jev-gc produced (`build_context`)
- `JevGC.stats()` -- how many spans were processed, kept HOT/WARM/COLD, and
  how many actually needed a Jev call vs. were resolved by the
  deterministic pre-filter alone

## Notebook

[`notebook.ipynb`](./notebook.ipynb) walks through the same scenario
step by step: running the agent, observing spans through jev-gc, and
comparing the naive "keep everything" context against jev-gc's
budget-fit one -- including the token counts behind the comparison in the
main [`README.md`](../../README.md#does-this-actually-matter).

## Notes

- This example is free-tier friendly by design: 4 turns, `gemini-2.0-flash`,
  and Jev at ~$0.042/M input tokens -- a full run costs a fraction of a
  cent.
- If `JEV_API_KEY` is unset, `build_jevgc()` falls back to
  `FakeJevClient` so the example still runs end-to-end with no network
  calls to TypeSafe (Gemini and Open-Meteo are still real).
