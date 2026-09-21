# Designing jev-gc's Jev questions

jev-gc asks Jev exactly four questions, all defined in `scorer.py`. This
doc records *why* they're shaped the way they are -- useful both for
extending jev-gc and as a worked example of question design against Jev's
`choice` / `score` / `noul` primitives.

## 1. `relevance_to_current_task` (Score)

Asked only for non-error, prefilter-ambiguous spans (Track A).

Real Jev `score` questions return an index into an ordered `criteria`
rubric, not a free `0.0-1.0` float (see `docs/architecture.md` for how this
differs from SPEC.md's original sketch). jev-gc uses a fixed 5-point
rubric (`scorer.RELEVANCE_RUBRIC`, low to high) rather than a per-call
custom one, for two reasons:

- **Consistency.** Every span is scored against the identical rubric, so
  `relevance_keep_threshold` in config means the same thing call to call.
- **Atomicity.** Per the "atomic questions, composed in code" pattern
  (docs.typesafe.ai), relevance is a single well-scoped gut-check --
  "how relevant, on this fixed ladder" -- not a compound judgment. If a
  host wants relevance weighted by other factors (recency, cost to
  re-fetch, etc.), that's a job for `policy.py`'s code, not a richer
  question.

The returned legend index is normalized by `client.py` to `[0.0, 1.0]`
before it ever reaches `policy.py`, so the rest of the codebase can treat
it as a continuous score per the frozen `JevScoreAnswer` contract.

## 2. `treatment` (Choice)

Asked alongside relevance (Track A). Options are exactly the four
`Treatment` enum values from `models.py`, each with a one-line description
as the Choice `criteria`. Kept deliberately separate from the relevance
Score question rather than merged into one compound "how relevant and how
should it be kept" question, per the same atomicity principle -- a low
relevance score and an `include_summary_only` treatment are two
independent judgments Jev can be *wrong* about independently, and
`policy.py` needs to reason about them independently too (see the
threshold interaction in `policy.DecisionPolicy._decide_success`).

## 3. `item_role` (Choice, optional)

Config-gated (`ask_item_role` on `JevBatchScorer`) because it doesn't
affect any tiering decision on its own -- SPEC.md §4.4 marks it explicitly
for "prioritization/dashboards", not the core keep/drop path. Every
`ItemRole` option **except** `uncertain` describes a specific structural
role; `uncertain` is the escape hatch SPEC.md §2 principle 4 requires for
every Choice question -- landing on it is a signal, not a normal outcome,
so `policy._role_of` treats an unparseable or missing role as `None`
rather than guessing.

## 4. `error_treatment` (Choice)

Asked only for errored spans (Track B), and it's the *only* question asked
about them -- no relevance Score, per SPEC.md §2 principle 5 ("only the
error's payload size is negotiable via Jev -- never its presence"). Notice
`ErrorTreatment` has no `drop` option at all: the type system itself
enforces that an errored span can never be evicted from the next prompt,
only compressed down to `keep_error_type_only`.

## General notes

- Every Choice question's `criteria` includes a short description per
  option (not just the bare enum value) -- Jev's docs recommend rich
  per-option descriptions over bare labels for more reliable
  classification, and it costs nothing extra (all questions in a request
  are still evaluated in parallel).
- No question in jev-gc asks Jev to do arithmetic, count things, or check
  an age/threshold -- those stay in `prefilter.py`/`policy.py` as plain
  code, per SPEC.md §2 principle 1 ("code calculates, Jev judges").
