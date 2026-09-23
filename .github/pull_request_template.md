# What and why

<!-- What changes, and what problem it solves. The diff says what; use this
     space for why. If it fixes an issue, link it. -->

## How this was verified

<!-- Not "tests pass" — say what you actually exercised. -->

- [ ] `ruff check .`
- [ ] `mypy src/jevgc` clean under `--strict`
- [ ] `pytest` passes, including the coverage gate
- [ ] New behaviour has tests, and they're deterministic (no network, no real sleeps)

## If this touches a decision path

<!-- Delete this section if it doesn't. -->

- [ ] Anything rule-decidable stayed in `prefilter.py` / `policy.py` rather than becoming a Jev question
- [ ] Every new Jev call has a code-owned fail-open fallback
- [ ] Errored spans can still never be dropped, only compressed
- [ ] Nothing is hard-deleted, or the exception is bounded and documented

## If this changes the published numbers

<!-- Delete this section if it doesn't. -->

- [ ] Regenerated `results/benchmark.json` with `--repeat 5` rather than editing figures by hand
- [ ] Re-ran `scripts/worked_example.py` and confirmed it's still byte-identical across runs
- [ ] Updated `docs/index.html` and `results/report.md` to match

## Anything reviewers should push back on

<!-- Shortcuts taken, things you weren't sure about, decisions worth a second
     opinion. This section being non-empty is a good sign, not a bad one. -->
