# jev-gc — Build Specification

**Audience:** Claude Code (or any implementer) building this repository out from
a bare-minimum skeleton. This document is the source of truth for architecture,
module contracts, coding standards, testing, and the parallel task breakdown.
The three files already in `src/jevgc/` (`models.py`, `exceptions.py`,
`__init__.py`) are **frozen contracts** — implement against them; don't redesign
them without a documented, deliberate reason recorded in a comment at the top of
the changed file.

---

## 1. What this project is

`jev-gc` is a Python library that plugs into an LLM agent's existing
OpenTelemetry instrumentation and performs **real-time context garbage
collection**: as the agent emits spans (tool calls, retriever calls, LLM
calls), jev-gc decides — continuously, not in a batch pass at the end —
what stays in the next prompt's context window, what gets compressed to a
pointer, and what gets archived.

The relevance/classification judgments that don't have an obvious
deterministic answer are delegated to **Jev**, TypeSafe AI's "System One"
model. Jev does not reason — it returns typed, probabilistic answers
(`Choice`, `Score`) to questions you define in advance, in ~70–500ms at
~$0.042 per million input tokens. Treat every design decision in this spec
as downstream of that one fact: **Jev classifies against a fixed option
set; it never invents a missing option, and it can be confidently wrong
when a situation doesn't fit the options it was given.** Every place Jev is
consulted must have a code-owned fail-open fallback.

### Non-goals

- jev-gc does **not** call an LLM to summarize content. An LLM may be used
  by the *host application* to synthesize a final answer from the context
  jev-gc assembles — that's out of scope for this library.
- jev-gc does **not** decide tool-calling logic, agent control flow, or
  retries. It is a side-effect-observing, context-assembling layer only.
- jev-gc never permanently deletes data. "Drop" means evicted from the
  next prompt, not destroyed — see the tiering model in §3.3.

---

## 2. Guiding principles (apply these when any spec detail is ambiguous)

1. **Code calculates. Jev judges.** Anything expressible as a threshold,
   age check, status check, or pin flag belongs in deterministic code
   (`prefilter.py`, `policy.py`), never in a Jev question. Jev is reserved
   for genuinely ambiguous relevance/role judgments on the remaining items.
2. **Fail open, not closed, by default — except where stated otherwise.**
   A false "keep" costs a few extra tokens. A false "drop" can silently
   break the agent's reasoning turns later with no error message. When Jev
   is unavailable, times out, or returns low confidence, default to
   keeping the item at its current tier.
3. **Never hard-delete.** Everything evicted from HOT context is still
   reachable — via WARM pointer + re-fetch, or via COLD archive retrieval.
4. **Every Choice question needs an escape hatch.** Never force Jev into
   a forced choice among options that might not fit the real situation —
   always include an "uncertain"/"none of these" style option, and treat
   landing on it as a signal to keep the item and let the human/host LLM
   sort it out, not as a normal classification outcome to route past.
5. **Errored spans are never dropped, only compressed.** A failed tool
   call is usually the *most* relevant thing in context even when its raw
   payload (e.g. a 4KB stack trace) is mostly noise. Only the error's
   *payload size* is negotiable via Jev — never its presence.
6. **Everything is batched.** Jev evaluates every question in a request in
   parallel at near-zero marginal cost — the scorer must never issue one
   HTTP call per span. Batch spans awaiting classification and fire one
   request per batch window.
7. **Determinism where it's cheap, probabilism where it's not.** Any
   function whose output should be reproducible for the same input
   (tests, audits) must not depend on live Jev calls without a seam to
   inject a fake/mock client.

---

## 3. Architecture

```
                         ┌─────────────────────────────────────────┐
                         │            Host Agent Process             │
                         │  (LangGraph / Strands / raw ReAct loop)    │
                         └───────────────┬─────────────────────────┘
                                          │ emits spans via
                                          │ opentelemetry-sdk
                                          ▼
                         ┌─────────────────────────────────────────┐
                         │        JevGCSpanProcessor (otel/)          │
                         │  on_end(span) → SpanRecord                 │
                         └───────────────┬─────────────────────────┘
                                          ▼
                         ┌─────────────────────────────────────────┐
                         │      Deterministic Pre-Filter              │
                         │      (prefilter.py)                        │
                         │  status==ERROR?        → always keep,      │
                         │                           route to error   │
                         │                           treatment path   │
                         │  referenced since?     → keep as-is        │
                         │  age>N turns AND        → drop, no Jev call│
                         │    status==OK AND                          │
                         │    zero-ish output?                        │
                         │  else                   → send to Jev      │
                         └───────────────┬─────────────────────────┘
                                          │ ambiguous spans, batched
                                          ▼
                         ┌─────────────────────────────────────────┐
                         │         JevBatchScorer (scorer.py)         │
                         │  batches on: N spans OR T milliseconds     │
                         │  (whichever first) → 1 Jev API call        │
                         │  Q1 (Score):  relevance_to_current_task    │
                         │  Q2 (Choice): treatment | error_treatment  │
                         │  Q3 (Choice): item_role (optional, for     │
                         │               prioritization/dashboards)   │
                         └───────────────┬─────────────────────────┘
                                          │ JevQuestionResult batch
                                          ▼
                         ┌─────────────────────────────────────────┐
                         │       Decision Policy (policy.py)          │
                         │  combines Jev answers + confidence +       │
                         │  recency + pin status + budget             │
                         │  → GCDecision (Tier, Treatment)            │
                         │  NEVER lets Jev's raw answer bypass         │
                         │  confidence thresholds or the fail-open    │
                         │  rule.                                     │
                         └───────────────┬─────────────────────────┘
                                          ▼
                         ┌─────────────────────────────────────────┐
                         │      Tiered Context Store (store.py)       │
                         │  HOT / WARM / COLD, pluggable backend       │
                         │  (in-memory default; SQLite/Redis          │
                         │  optional backends)                        │
                         └───────────────┬─────────────────────────┘
                                          │ gc.build_context(task, budget)
                                          ▼
                         ┌─────────────────────────────────────────┐
                         │   Budget Allocator + Context Builder       │
                         │   (policy.py + context_builder.py)         │
                         │   greedily fills token budget by            │
                         │   relevance; errored spans always win a    │
                         │   slot; renders final prompt-ready text    │
                         └─────────────────────────────────────────┘
```

### 3.1 Data flow contract

Every module boundary above passes one of the types already defined in
`src/jevgc/models.py`:

- OTel → pre-filter: `SpanRecord`
- pre-filter → scorer: `SpanRecord` (only the ambiguous subset)
- scorer → policy: `JevQuestionResult` (batched, keyed by `item_id`)
- policy → store: `GCDecision`
- store → context builder: `ContextItem`

Do not introduce new cross-module types without adding them to `models.py`
first and documenting why the existing types don't cover the case.

### 3.2 Package layout

```
src/jevgc/
  __init__.py            # public API surface (already scaffolded)
  models.py               # FROZEN — shared data contracts
  exceptions.py            # FROZEN — exception hierarchy
  config.py                # JevGCConfig (pydantic-settings): env + YAML
  gc.py                    # JevGC facade — the class users actually touch
  prefilter.py              # deterministic pre-filter (§3, principle 1)
  scorer.py                 # JevBatchScorer — batching + Jev API calls
  policy.py                 # DecisionPolicy + BudgetAllocator
  store.py                   # TieredContextStore + Backend protocol
  context_builder.py         # assembles final rendered prompt context
  telemetry.py                # jev-gc's own OTel metrics (self-observability)
  jev_client/
    __init__.py
    client.py                 # JevClient protocol + HTTPJevClient impl
    fakes.py                   # FakeJevClient for tests/examples (no network)
  otel/
    __init__.py
    processor.py                # JevGCSpanProcessor(SpanProcessor)
    translate.py                 # ReadableSpan → SpanRecord
    attributes.py                 # GenAI semantic-convention constants
  backends/
    __init__.py
    memory.py                     # InMemoryBackend (default)
    sqlite.py                      # optional persistent backend
  integrations/
    __init__.py
    langgraph.py                    # LangGraph node/hook wrapper
    strands.py                       # Strands Agents hook (optional dep)
tests/
  unit/                              # one test module per src module, mirrored
  integration/                        # end-to-end: fake OTel spans → context
  fixtures/                            # shared pytest fixtures, sample spans
examples/
  strands_dummy_agent/
    dummy_agent.py                    # runnable Strands agent emitting OTel
    notebook.ipynb                     # walks through wiring jev-gc to it
    README.md
docs/
  architecture.md
  quickstart.md
  jev_question_design.md               # how to write good Choice/Score schemas
.github/workflows/
  ci.yml
scripts/
  setup.sh                              # one-shot dev env bootstrap (uv/pip)
  ci-local.sh                            # mirrors ci.yml checks locally
```

---

## 4. Module specs

### 4.1 `config.py`

`JevGCConfig` (pydantic-settings `BaseSettings`), loadable from:
- a YAML file (`JevGC.from_config("jevgc.yaml")`)
- environment variables, prefix `JEVGC_` (e.g. `JEVGC_JEV_API_KEY`)
- explicit kwargs, for tests

Required fields:

```yaml
jev:
  api_key: ${JEV_API_KEY}          # required; never logged, never repr'd
  base_url: https://api.typesafe.ai/v1
  timeout_seconds: 2.0
  max_retries: 2

prefilter:
  keep_last_n_turns: 3              # always HOT regardless of score
  drop_zero_output_after_turns: 10  # pure code, never hits Jev

scorer:
  batch_max_size: 50
  batch_max_wait_ms: 200

policy:
  relevance_keep_threshold: 0.35
  min_confidence_to_drop: 0.6        # below this, fail open (keep)
  error_default_treatment: keep_error_summary_only

store:
  backend: memory                     # memory | sqlite
  sqlite_path: null

telemetry:
  emit_self_metrics: true             # tokens saved, spans dropped, etc.
```

`api_key` must use pydantic's `SecretStr`. `ConfigurationError` (from
`exceptions.py`) is raised at load time if `api_key` is missing —
this is the one place fail-fast, not fail-open, is correct (§2).

### 4.2 `jev_client/client.py`

Define a `Protocol` (not an ABC — keep it structurally typed so
`FakeJevClient` in tests needs no inheritance):

```python
class JevClient(Protocol):
    async def batch_ask(
        self, requests: list[JevBatchRequest]
    ) -> list[JevQuestionResult]: ...
```

`HTTPJevClient` implements this over `httpx.AsyncClient`, with:
- retry via `tenacity` (exponential backoff + jitter) on `JevRateLimitError`
  and transient network errors, capped at `config.jev.max_retries`
- `JevTimeoutError` raised on timeout, `JevAPIError` on non-2xx/malformed body
- **never** retries on a well-formed error response — only on
  timeout/rate-limit/connection failure
- request/response validated against pydantic models (no raw dict passing)

`fakes.py` provides `FakeJevClient` — takes a canned response map or a
callable `(requests) -> results`, used by every unit test that touches
the scorer or policy, and by the Strands example so it can run without a
real Jev API key.

### 4.3 `prefilter.py`

Pure functions, no I/O, no Jev dependency. Given a `SpanRecord` and the
running context state (turn index, reference counts, pin set), returns
one of:
- `PrefilterResult.KEEP` (short-circuits — never sent to Jev)
- `PrefilterResult.DROP` (short-circuits — never sent to Jev)
- `PrefilterResult.AMBIGUOUS` (forwarded to the scorer)

Rules, in order (first match wins):
1. Span is pinned (system prompt, explicit user constraint, open TODO) → `KEEP`
2. `span.is_error` → `KEEP` (routes to error-treatment path, not the
   normal relevance path — see §4.4)
3. Referenced by a later span/turn → `KEEP`
4. Within `keep_last_n_turns` of current turn → `KEEP`
5. Age > `drop_zero_output_after_turns` AND status OK AND output is
   empty/trivial (e.g. `output_token_count == 0` or below a tiny
   constant) → `DROP`
6. Otherwise → `AMBIGUOUS`

100% unit-testable without mocks — this is the highest-leverage module
for test coverage since it's pure logic.

### 4.4 `scorer.py`

`JevBatchScorer` owns batching. Two independent question tracks:

**Track A — successful spans (relevance/treatment):**
- Q1 `Score`: `relevance_to_current_task` (0.0–1.0) + confidence
- Q2 `Choice`: `treatment` ∈ `Treatment` enum (include_full /
  include_summary_only / keep_pointer_only / drop)
- Q3 `Choice` (optional, config-gated): `item_role` ∈ `ItemRole` enum

**Track B — errored spans (error treatment only, no relevance question —
principle 5):**
- Q1 `Choice`: `error_treatment` ∈ `ErrorTreatment` enum

Batching policy: accumulate `AMBIGUOUS` spans until either
`batch_max_size` is reached or `batch_max_wait_ms` elapses since the
first item in the current batch — whichever comes first. Use an
`asyncio.Task`-based timer, not a blocking sleep, so it doesn't stall the
agent loop.

On `JevTimeoutError`/`JevAPIError` for a batch: log, emit a telemetry
counter, and return fail-open results for every item in that batch
(`relevance=None, confidence=0.0`) — the policy module's job is to turn
that into "keep at current tier."

### 4.5 `policy.py`

`DecisionPolicy.decide(span, jev_result) -> GCDecision`:

```
if jev_result is None (fail-open path):
    return GCDecision(tier=HOT-or-current, reason="jev_unavailable_fail_open", used_jev=False)

if span.is_error:
    treatment = jev_result.choice.option  # ErrorTreatment
    tier = WARM if treatment != KEEP_FULL_TRACE else HOT
    return GCDecision(..., used_jev=True)

if jev_result.score.confidence < config.policy.min_confidence_to_drop:
    return GCDecision(tier=HOT, reason="low_confidence_fail_open", used_jev=True)

if jev_result.score.score < config.policy.relevance_keep_threshold:
    tier = WARM or COLD based on jev_result.choice.option (Treatment.DROP → COLD via store TTL, not immediate)
else:
    tier = HOT
```

`BudgetAllocator.allocate(items: list[ContextItem], budget_tokens: int) ->
list[ContextItem]`: greedy knapsack by `relevance_score` descending,
**errored-span items always included first regardless of score** (§2
principle 5), then fill remaining budget by descending relevance until
exhausted. Must be a pure function — fully unit-testable with synthetic
`ContextItem` lists, no Jev/store dependency.

### 4.6 `store.py`

`Backend` protocol: `get`, `put`, `list_by_tier`, `move_tier`. Default
`InMemoryBackend` (dict-based, fine for single-process agents). Optional
`SQLiteBackend` for persistence across process restarts (implement only
if time allows — mark clearly as "nice to have" in the PR, core spec
requirement is `InMemoryBackend` + the protocol).

`TieredContextStore` wraps a `Backend` and is the only module allowed to
change an item's `Tier` post-decision (e.g., re-promoting a WARM item to
HOT if a later span references its `span_id` — this is a deterministic
rule, not a Jev call).

### 4.7 `otel/processor.py` + `otel/translate.py`

`JevGCSpanProcessor` implements OpenTelemetry's `SpanProcessor` interface
(`on_start`, `on_end`, `shutdown`, `force_flush`). `on_end` must:
1. Translate `ReadableSpan` → `SpanRecord` via `translate.py`
2. Never raise into the SDK's export pipeline — wrap in try/except,
   log + emit telemetry counter on translation failure
   (`OTelIntegrationError`), and skip the span rather than crash
3. Hand the `SpanRecord` to the pre-filter → scorer → policy → store
   pipeline asynchronously (don't block span export on Jev latency)

`translate.py` extracts GenAI semantic-convention attributes
(`gen_ai.operation.name`, `gen_ai.tool.name`, error attributes) where
present — see `otel/attributes.py` for the constant names, sourced from
the OpenTelemetry Semantic Conventions for Generative AI spec. Must
degrade gracefully for spans that don't carry these (e.g. a raw DB call).

### 4.8 `gc.py` — the public facade

```python
class JevGC:
    @classmethod
    def from_config(cls, path: str | Path) -> "JevGC": ...

    def attach_to_tracer_provider(self, provider: TracerProvider) -> None: ...

    def build_context(self, *, task: str, budget_tokens: int) -> str: ...

    def stats(self) -> GCStats:  # tokens saved, spans processed/dropped, Jev call count/latency
        ...
```

This is the only class most users touch. Keep its surface small and
stable — everything else is an implementation detail behind it.

### 4.9 `integrations/langgraph.py`

A `LangGraph` node factory: `jev_gc_node(gc: JevGC) -> Callable[[State],
State]` that reads the graph's message/tool-result history from state,
lets jev-gc's OTel processor observe it (or, if the host isn't emitting
OTel spans, exposes a manual `gc.observe(span_record)` escape hatch), and
returns a trimmed context ready for the next LLM call. Document the
"manual observe" path clearly since not every LangGraph app is
already instrumented with OTel.

### 4.10 `integrations/strands.py`

Strands Agents has native OpenTelemetry instrumentation (GenAI semantic
conventions) out of the box. This integration should be closer to a thin
convenience wrapper than new logic: a helper that pulls a Strands agent's
configured `TracerProvider` and calls `gc.attach_to_tracer_provider(...)`
on it, plus a documented gotcha list (e.g. any Strands-specific span
naming to be aware of when tuning `otel/translate.py`).

---

## 5. Jev API integration details

Model the request/response shapes conservatively (Jev's public API surface
is still young) — the concrete HTTP contract is the one thing about this
spec Claude Code should treat as **provisional**: verify current field
names/endpoint against TypeSafe's docs (docs.typesafe.ai) before finalizing
`jev_client/client.py`, and isolate this in `client.py` alone so a schema
change never ripples past that one file (this is exactly why `JevClient` is
a `Protocol` — everything else in the codebase depends on the protocol, not
on `HTTPJevClient`).

Expected shape, per TypeSafe's documented primitives:

```json
// Request
{
  "state": "current_task: ...\nspan: ...",
  "questions": [
    {"id": "q1", "type": "score", "prompt": "relevance_to_current_task"},
    {"id": "q2", "type": "choice", "prompt": "treatment",
     "options": ["include_full", "include_summary_only", "keep_pointer_only", "drop"]}
  ]
}

// Response
{
  "answers": [
    {"question_id": "q1", "score": 0.12, "confidence": 0.79},
    {"question_id": "q2", "option": "keep_pointer_only", "probability": 0.68}
  ]
}
```

`scorer.py` packs **all ambiguous spans in one batch window into a single
Jev call** by giving each span's questions a unique `item_id`-prefixed
question id (e.g. `"{item_id}::q1"`) and splitting the response back out —
this is what makes batching cheap (Jev evaluates every question in a
request in parallel).

---

## 6. Coding standards

- **Python ≥3.10**, fully type-hinted, `mypy --strict` clean (see
  `pyproject.toml`, already configured).
- **Pydantic v2** for every data model and for config — no raw dicts
  crossing module boundaries.
- **`async` for anything that calls Jev or does I/O**; pure logic
  (`prefilter.py`, `policy.py`'s decision function, `BudgetAllocator`)
  stays synchronous and dependency-free so it's trivially testable.
- **`ruff`** for lint + format (config already in `pyproject.toml`); no
  bare `except:`, no mutable default args, no wildcard imports.
- Every public class/function gets a docstring stating **why**, not just
  what — this repo is meant to be read by people evaluating whether to
  trust Jev in their own agent's loop; the reasoning is the product.
- No global mutable state. `JevGC` instances must be safely constructible
  more than once per process (tests instantiate many).
- Logging via stdlib `logging`, never `print`. One logger per module
  (`logging.getLogger(__name__)`).
- Secrets (`api_key`) must never appear in `repr()`, logs, or exception
  messages — use `pydantic.SecretStr` and check this explicitly in a test.

---

## 7. Testing requirements

- **Unit tests** (`tests/unit/`, mirrors `src/jevgc/` structure 1:1):
  - `prefilter.py`: table-driven tests covering every rule branch above,
    including the ordering (pin beats age, error beats recency, etc.)
  - `policy.py`: fail-open on `None` Jev result, fail-open on low
    confidence, correct tier assignment per `Treatment`/`ErrorTreatment`
  - `scorer.py`: batching triggers on size AND on timeout (use
    `freezegun`/`asyncio` test clock — no real sleeps in tests), retry
    behavior via `FakeJevClient` raising configured errors, fail-open
    result shape on exhausted retries
  - `jev_client/client.py`: HTTP behavior mocked via `respx`; test 2xx
    happy path, 429 retry-then-succeed, 429 retry-exhausted →
    `JevRateLimitError`, malformed body → `JevAPIError`, timeout →
    `JevTimeoutError`
  - `store.py`: tier transitions, `InMemoryBackend` CRUD
  - `otel/translate.py`: `ReadableSpan` fixtures (both GenAI-convention
    and plain spans) → correct `SpanRecord`, malformed span → skipped,
    not raised
  - Secrets-never-leak test: assert `str(config)` / `repr(config)` /
    raised exceptions never contain the raw API key
- **Integration tests** (`tests/integration/`): synthetic span sequences
  fed through the full pipeline (`FakeJevClient` — no network) asserting
  on the final assembled context string/token count for realistic
  scenarios (a table-routing-style investigation with one dead end, one
  error, one duplicate).
- **Coverage gate:** ≥85% (already set in `pyproject.toml`,
  `fail_under = 85`), enforced in CI.
- Every test must be deterministic — no real `time.sleep`, no real
  network calls, no reliance on real Jev API availability. The full
  suite should run in well under 30 seconds.

---

## 8. CI (`.github/workflows/ci.yml`)

On every push/PR: matrix over Python 3.10/3.11/3.12, run:
1. `ruff check .`
2. `mypy src/jevgc`
3. `pytest` with coverage, fail under 85%
4. Upload coverage artifact

Keep it to one workflow file, no unnecessary complexity.

---

## 9. Example: Strands dummy agent (`examples/strands_dummy_agent/`)

**Purpose:** a small, runnable agent (using the
[Strands Agents](https://strandsagents.com) SDK) that does a multi-step
tool-calling task — reuse the invoice/table-routing scenario from the
design discussion (`query_erp_table`, `query_warehouse_shipments`,
`query_shipping_carrier_api`, deliberately including one tool call that
errors and one that returns an irrelevant result) — purely to **emit
realistic OpenTelemetry spans** for jev-gc to consume. The agent itself
is not the point; it exists to prove the integration works end to end.

Deliverables:
- `dummy_agent.py`: Strands agent + 3–4 mock tools (no real external
  calls — fabricate plausible data so the example needs no credentials
  beyond a Jev API key, and works with `FakeJevClient` if that's not
  set either)
- `notebook.ipynb`: narrated walkthrough —
  1. Run the agent without jev-gc, show the raw growing context/token
     count turn by turn
  2. Attach `JevGC`, re-run, show the same session's token count with
     GC active, and print a few `GCDecision`s with their `reason` field
     so the reader can see *why* each item was kept/dropped
  3. Show the final assembled context string
- `README.md`: how to run it, what it demonstrates, link back to the
  main `SPEC.md` for the "why"

This notebook is the primary artifact people will actually look at to
decide whether to trust jev-gc — prioritize clarity over cleverness.

---

## 10. Suggested parallel task breakdown

If building with multiple subagents concurrently, this split minimizes
file overlap (each agent owns disjoint files) while respecting the
dependency that everything imports `models.py`/`exceptions.py`, which are
already frozen and provided:

| Agent | Owns | Depends on |
|---|---|---|
| **A — Core decision engine** | `config.py`, `prefilter.py`, `policy.py`, their unit tests | `models.py`, `exceptions.py` only |
| **B — Jev client + scorer** | `jev_client/` (protocol, HTTP impl, fakes), `scorer.py`, their unit tests | `models.py`, `exceptions.py` only |
| **C — Store + OTel plumbing** | `store.py`, `backends/`, `otel/` (processor, translate, attributes), their unit tests | `models.py`, `exceptions.py` only |
| **D — Facade + integrations** | `gc.py`, `context_builder.py`, `telemetry.py`, `integrations/`, integration tests | **Runs after A/B/C land** — this is the assembly layer |
| **E — Example + docs** | `examples/strands_dummy_agent/`, `docs/`, root `README.md` polish | `gc.py`'s public API (can stub against the facade signature in §4.8 and start early) |

Agents A, B, C, and E can start immediately in parallel — they share only
the already-frozen `models.py`/`exceptions.py` contracts. Agent D is the
integration point and should start once A/B/C's public functions/classes
exist (even as thin stubs with correct signatures), then wire them
together and add the end-to-end integration tests. Run the full test
suite + `mypy --strict` + `ruff` after D completes as the final gate
before considering the build done.

---

## 11. Definition of done

- [ ] All modules in §3.2 implemented per their contracts in §4
- [ ] `pytest` green, ≥85% coverage
- [ ] `mypy --strict` clean
- [ ] `ruff check .` clean
- [ ] Strands example runs end-to-end and its notebook executes top to
      bottom without errors
- [ ] `README.md` quickstart snippet actually runs as written
- [ ] No secret ever appears in a log line, exception message, or repr
      (verified by a test, not just inspection)
