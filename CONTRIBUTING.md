# Contributing to jev-gc

Thanks for taking the time. This file covers the practical setup; the
*reasoning* behind the architecture lives in [`SPEC.md`](./SPEC.md), and
reading §2 ("Guiding principles") before proposing a design change will save
us both a round trip.

## Setup

```bash
git clone https://github.com/adigulalkari/Jev_GC
cd Jev_GC
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,langgraph,strands]"
```

No API key is needed to develop or run the tests — `FakeJevClient` covers
every path. A real `JEV_API_KEY` is only required for
`scripts/relevance_demo.py` and the bundled example agent.

## The gate

CI runs these three on Python 3.10, 3.11 and 3.12. Run them locally first:

```bash
./scripts/ci-local.sh
```

or individually:

```bash
ruff check .          # lint + format
mypy src/jevgc        # must be --strict clean
pytest                # coverage gate enforced via pyproject
```

Pull requests that fail any of the three won't be merged, and that includes
`mypy --strict` — this library is meant to be embedded in other people's agent
loops, so the types are load-bearing rather than decorative.

## Things that will come up in review

These aren't style nits; each one is a decision the project has already made
deliberately, and changing one needs an argument rather than a patch.

- **Code calculates, Jev judges.** Anything expressible as a threshold, age
  check, status check or pin flag belongs in `prefilter.py` or `policy.py`, not
  in a Jev question. Adding a Jev call for something a rule could decide will
  be sent back.
- **Fail open.** Every place Jev is consulted needs a code-owned fallback that
  keeps the item. A false keep costs a few tokens; a false drop breaks an
  agent's reasoning several turns later with no error and no trace.
- **Errored spans are never dropped.** Only their payload size is negotiable.
- **Never hard-delete.** Evicted means out of the next prompt, not destroyed.
  The one bounded exception is the archive's memory ceiling, and it is
  documented at length in `archive.py` — extend that pattern rather than adding
  new silent deletions.
- **`models.py` and `exceptions.py` are frozen contracts.** New cross-module
  types go in the module that owns them (see the header comment in
  `archive.py` for a worked precedent) unless you have a documented reason,
  recorded in a comment, to widen the frozen set.
- **Docstrings state *why*, not what.** The reasoning is the product here —
  people read this repo to decide whether to trust a probabilistic model inside
  their own agent loop. A docstring that only restates the signature will be
  flagged.
- **No `print`, no global mutable state, no bare `except`.** `JevGC` must be
  constructible many times per process; the tests do exactly that.

## Tests

`tests/unit/` mirrors `src/jevgc/` one to one; `tests/integration/` drives the
whole pipeline with a fake client. Every test must be deterministic — no real
sleeps, no network, no dependence on a live Jev API. The full suite should stay
well under 30 seconds.

If you change anything that affects the published numbers, regenerate them
rather than editing them by hand:

```bash
python scripts/benchmark_overhead.py --repeat 5 --json > results/benchmark.json
python scripts/worked_example.py
```

## Commits and pull requests

Explain *why* in the commit message, not just what — the diff already says
what. Keep one logical change per PR where you reasonably can.

## Reporting bugs

Open an issue using the bug template. The single most useful thing you can
include is the `reason` string from the `GCDecision` that surprised you, plus
the relevant `prefilter`/`policy` settings, since together they usually explain
the behaviour outright.
