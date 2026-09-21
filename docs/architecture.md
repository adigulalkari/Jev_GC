# Architecture

This is the implementation-level companion to [`SPEC.md`](../SPEC.md); read
that first for the full design rationale. This document tracks how the
pipeline actually behaves as built, module by module.

## Pipeline

```
OTel span ends
     │
     ▼
JevGCSpanProcessor.on_end          (otel/processor.py, otel/translate.py)
     │  ReadableSpan -> SpanRecord; never raises, skips on translation failure
     ▼
JevGC.observe(span)                (gc.py)
     │
     ▼
apply_prefilter(span, ctx)         (prefilter.py)
     │
     ├── DROP ──────────────────────────────────────────────┐
     ├── KEEP, not span.is_error ──────────────────────────┐ │
     │                                                      │ │
     └── AMBIGUOUS, or KEEP-and-is_error                    │ │
              │                                             │ │
              ▼                                             │ │
     JevBatchScorer.submit(span)   (scorer.py)               │ │
              │ batched on size or time, one Jev API call     │ │
              ▼                                             │ │
     DecisionPolicy.decide(span, jev_result)  (policy.py)     │ │
              │                                             │ │
              ▼                                             ▼ ▼
                        GCDecision (models.py, frozen contract)
                                   │
                                   ▼
                build_context_item(span, decision)  (context_builder.py)
                                   │
                                   ▼
                    TieredContextStore.put(item)     (store.py)


JevGC.build_context(task, budget_tokens)
     │
     ▼
BudgetAllocator.allocate(HOT + WARM items, budget_tokens)  (policy.py)
     │  errored-span items always included first, then greedy by relevance
     ▼
assemble_context(selected)         (context_builder.py)
     │
     ▼
prompt-ready string
```

## Why error spans still reach the scorer

SPEC.md §4.3 lists `PrefilterResult.KEEP` as short-circuiting (never sent to
Jev). But §2 principle 5 and §4.4's Track B require errored spans to still
get a Jev-driven *payload size* decision (full trace / summary / type-only)
-- only their *presence* is deterministic, not their size.

`JevGC.observe` resolves this by treating "prefilter KEEP because the span
errored" differently from "prefilter KEEP for any other reason" (pin,
reference, recency): only the latter short-circuits before Jev. Any errored
span -- regardless of which prefilter rule matched first -- is routed to
`JevBatchScorer`, which asks only the `error_treatment` Choice question
(never `relevance`, per principle 5) and never lets a DROP-shaped answer
come back, since `ErrorTreatment` has no drop option at all.

## Why `item_id == span_id`

`SpanRecord.span_id` is already unique within a process (OTel span IDs are
effectively unique, and hosts using the manual `observe` escape hatch are
expected to mint unique ids too). Reusing it as the scorer's `item_id`
avoids an extra id allocation and makes cross-referencing a `GCDecision`
back to the `FakeJevClient` call that produced it trivial in tests and logs.

## The real Jev API vs. SPEC.md's sketch

SPEC.md §5 explicitly flags its request/response shapes as provisional and
scopes any deviation to `jev_client/client.py` alone. The real API
(verified against `docs.typesafe.ai` / `api.typesafe.ai`) differs in three
ways that are fully contained to that file:

1. `questions` is a dict keyed by question id, not a list.
2. `choice` questions take `criteria: dict[option, description]`; `score`
   questions take an ordered `criteria: list[str]` rubric and return an
   index into it, not a raw `0.0-1.0` float.
3. Exactly one `state` string is accepted per API call.

Consequences, both isolated to `client.py`:

- `relevance_to_current_task` is asked as a `score` question against a
  fixed 5-point rubric (`RELEVANCE_RUBRIC`); the returned legend index is
  normalized to `[0.0, 1.0]` before being wrapped in the frozen
  `JevScoreAnswer` contract.
- To batch N spans' distinct content into one HTTP call despite the
  single-`state` constraint, each request's `state` is rendered under a
  `=== ITEM <item_id> ===` heading inside one composite document, and every
  question id is prefixed `"{item_id}::"` -- exactly the scheme SPEC.md
  already specifies for splitting the response back out.

See `docs/jev_question_design.md` for how the Choice/Score schemas
themselves were designed.
