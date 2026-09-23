"""One investigation, end to end, with every jev-gc decision shown.

WHY THIS EXISTS
---------------
Aggregate numbers (see `scripts/benchmark_overhead.py`) tell a reader how much
jev-gc saves, but not whether the savings were *safe*. The only way to answer
that is to show one concrete session in full: what each span contained, what
jev-gc decided about it, and -- crucially -- the `reason` string behind the
decision. A reader who can audit one case has grounds to believe the table; a
reader given only the table has to take it on faith.

So this script is deliberately narrow. It runs a single realistic multi-turn
bug investigation against *this repository*, prints every decision with its
reason, then demonstrates the two claims that are hardest to believe without
seeing them: that an evicted span is genuinely gone from the prompt yet still
discoverable by keyword and restorable verbatim, and that jev-gc can be held to
account afterwards for evictions that cost the agent something.

WHAT IS REAL AND WHAT IS SIMULATED
----------------------------------
Real: the tool output (actual `git grep`, actual file line-ranges, an actual
`git log` line, and an actual `FileNotFoundError` from an actual failed read,
all against this repo), the whole jev-gc pipeline (prefilter -> scorer ->
policy -> store -> archive -> context builder), the token accounting, and every
`reason` string.

Simulated: Jev's answers. They come from a `FakeJevClient` driven by a fixed
scripted responder -- no network, no API key. That makes the run reproducible,
but it also means this script demonstrates the *mechanism*, not the quality of
Jev's judgment. Nothing here is evidence that Jev scores these spans the way
the script says it does. For that question see `scripts/relevance_demo.py`,
which calls the real API.

Two of the eight spans are scenario scaffolding rather than tool output: the
pinned host constraint (which is the task statement) and a zero-output cache
probe (present so the prefilter's DROP rule is visible). Both are labelled as
such in the output.

Run:
    .venv/bin/python scripts/worked_example.py
    .venv/bin/python scripts/worked_example.py --json

Deterministic: no network, no clock reads in the span fixtures, and a fixed
scripted responder (SPEC.md §2 principle 7). Re-running at the same commit
reproduces this output byte for byte; the repo's HEAD sha is printed in the
header so a reader knows which snapshot produced the text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jevgc.archive import ColdIndexEntry
from jevgc.config import JevGCConfig
from jevgc.context_builder import estimate_tokens
from jevgc.gc import JevGC
from jevgc.jev_client.client import JevBatchRequest
from jevgc.jev_client.fakes import FakeJevClient
from jevgc.models import (
    ErrorTreatment,
    GCDecision,
    JevChoiceAnswer,
    JevQuestionResult,
    JevScoreAnswer,
    SpanRecord,
    SpanStatus,
    Treatment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBPROCESS_TIMEOUT_S = 30

#: Printable width. Kept under 100 so the transcript pastes into a `<pre>`
#: block without horizontal scrolling on a normal page column.
WIDTH = 98

TASK = (
    "Bug report: build_context() can silently drop ALL errored spans when their "
    "combined token count exceeds the budget. Is that actually possible?"
)
BUDGET_TOKENS = 900

#: What the agent types three turns later, after the investigation has moved on
#: from BudgetAllocator to a different question entirely: why did a rehydrate
#: come back without content? It half-remembers reading something about this.
SEARCH_QUERY = "archive released max_content_bytes discoverability"

#: A string that appears only in the evicted span. Used as the before/after
#: probe in section 3 -- if it is absent from the assembled context, the
#: eviction is real and not cosmetic.
EVICTION_MARKER = "max_content_bytes"

# The two final answers a shadow replay produces (SPEC.md's regret method:
# run the task twice, once with full context and once with jev-gc evicting,
# then diff the answers). Written out here rather than generated, because
# generating them would need a real LLM and this script makes no network call.
BASELINE_OUTPUT = (
    "No: BudgetAllocator appends every errored item to `selected` before the "
    "remaining-budget loop runs (policy.py:151-153), so an error span cannot be "
    "skipped -- the budget simply goes negative. The one place an evicted span "
    "really does become unrecoverable is the archive's own ceiling: past "
    "max_content_bytes the least-recently-used entries have their full text "
    "released while their keywords are kept, which preserves discoverability "
    "but not verbatim restoration."
)
EVICTED_OUTPUT = (
    "No: BudgetAllocator appends every errored item to `selected` before the "
    "remaining-budget loop runs (policy.py:151-153), so an error span cannot be "
    "skipped -- the budget simply goes negative. I found nothing else in the "
    "eviction path that would make a span unrecoverable."
)

# Question ids the scorer puts on the wire. Mirrored as literals rather than
# imported because `scorer._Q_*` are private to that module -- the same thing
# tests/integration/test_pipeline.py and scripts/benchmark_overhead.py do.
Q_RELEVANCE = "relevance"
Q_ERROR_TREATMENT = "error_treatment"

#: Scripted Jev answers: span_id -> (relevance, confidence, treatment). Fixed
#: values, not sampled ones, so two runs exercise identical policy branches.
JEV_ANSWERS: dict[str, tuple[float, float, str]] = {
    "grep-principle-5": (0.91, 0.93, Treatment.INCLUDE_FULL.value),
    "read-budget-allocator": (0.94, 0.95, Treatment.INCLUDE_FULL.value),
    "read-archive-budget": (0.07, 0.88, Treatment.DROP.value),
    "grep-principle-5-again": (0.21, 0.90, Treatment.KEEP_POINTER_ONLY.value),
    "gitlog-policy": (0.58, 0.84, Treatment.INCLUDE_FULL.value),
}
#: Scripted Jev answers for errored spans: span_id -> (error treatment, prob).
#: Note every option here keeps the span; SPEC.md §2 principle 5 makes only the
#: payload size negotiable, never the span's presence.
JEV_ERROR_ANSWERS: dict[str, tuple[str, float]] = {
    "read-missing-module": (ErrorTreatment.KEEP_ERROR_SUMMARY_ONLY.value, 0.86),
}


# --------------------------------------------------------------------------
# Real tools, run against this repository
# --------------------------------------------------------------------------


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run git in the repo root with a fixed locale.

    `LC_ALL=C` is pinned because this script's whole value is that its output
    can be diffed against a published transcript; a locale-dependent collation
    or message string would make two correct runs disagree.
    """
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
        env={**os.environ, "LC_ALL": "C"},
        check=False,
    )


def tool_git_grep(pattern: str, *paths: str) -> str:
    """`git grep -n -i` for a literal pattern, as the triage agent's tool does."""
    result = _run_git(["grep", "-n", "-i", "--", pattern, *paths])
    if result.returncode == 1:
        return f"No matches for {pattern!r}."
    if result.returncode != 0:
        raise RuntimeError(f"git grep failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def tool_read_file(path: str, start_line: int, end_line: int) -> str:
    """Read an inclusive, 1-indexed line range from a file in this repo.

    Raises with the *repo-relative* path rather than the absolute one so the
    error span's rendered text is identical on every machine -- an absolute
    path would make the published transcript unreproducible for anyone who
    checked the repo out somewhere else.
    """
    target = REPO_ROOT / path
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {path!r}")
    lines = target.read_text().splitlines()
    excerpt = lines[max(start_line - 1, 0) : end_line]
    numbered = [f"{i}: {line}" for i, line in enumerate(excerpt, start=start_line)]
    return "\n".join(numbered)


def tool_git_log(path: str, max_commits: int) -> str:
    """Most recent commits touching `path`, short-sha + date + subject."""
    result = _run_git(
        ["log", f"-n{max_commits}", "--pretty=format:%h %ad %s", "--date=short", "--", path]
    )
    if result.returncode != 0:
        raise RuntimeError(f"git log failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def repo_head() -> str:
    result = _run_git(["rev-parse", "--short", "HEAD"])
    return result.stdout.strip() or "(unknown)"


# --------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptedSpan:
    """One span plus the narration a reader needs to judge the decision.

    `tool` and `why` exist only for the transcript: `tool` is the call that
    produced the content (so the reader can re-run it), and `why` says which
    decision path this span is in the scenario to exercise. Neither is visible
    to jev-gc.
    """

    span: SpanRecord
    tool: str
    why: str
    pinned: bool = False


def _span(
    span_id: str,
    name: str,
    turn_index: int,
    *,
    index: int,
    input_preview: str | None = None,
    output_preview: str | None = None,
    output_token_count: int = 0,
    status: SpanStatus = SpanStatus.OK,
    error_type: str | None = None,
    error_message: str | None = None,
) -> SpanRecord:
    """Build a SpanRecord with synthetic but fixed timestamps.

    Timestamps are derived from `index` rather than read from the clock: a real
    OTel span carries wall-clock nanoseconds, which would change every run and
    is the one field in this fixture that nothing downstream reads (jev-gc uses
    `turn_index` for recency, not timestamps).
    """
    start_ns = index * 1_000_000_000
    return SpanRecord(
        span_id=span_id,
        trace_id="worked-example-trace",
        name=name,
        status=status,
        start_time_unix_ns=start_ns,
        end_time_unix_ns=start_ns + 40_000_000,
        turn_index=turn_index,
        input_preview=input_preview,
        output_preview=output_preview,
        output_token_count=output_token_count,
        error_type=error_type,
        error_message=error_message,
    )


def _failed_read() -> tuple[str, str]:
    """Actually attempt a read that cannot succeed, and return its real error.

    Fabricating an error type and message would be easier, but the point of
    this script is that the payloads are real. If the file ever exists, the
    scenario is silently wrong, so that case raises rather than falling back.
    """
    missing = "src/jevgc/nonexistent_module.py"
    try:
        tool_read_file(missing, 1, 40)
    except FileNotFoundError as exc:
        return type(exc).__name__, str(exc)
    raise RuntimeError(f"{missing} unexpectedly exists; the error-span scenario is invalid")


def build_session() -> list[ScriptedSpan]:
    """The investigation, as a list of spans in the order the agent emitted them.

    Shaped so that every decision path jev-gc can take is visible exactly once:
    a deterministic prefilter KEEP, a deterministic prefilter DROP, two Jev
    keeps, a Jev eviction to COLD, a Jev demotion to WARM, and an errored span
    that gets compressed but never dropped.
    """
    grep_pattern = "principle 5"
    grep_all = tool_git_grep(grep_pattern, "src/jevgc")
    grep_narrow = tool_git_grep(grep_pattern, "src/jevgc/policy.py", "src/jevgc/scorer.py")
    allocator_body = tool_read_file("src/jevgc/policy.py", 148, 156)
    archive_budget = tool_read_file("src/jevgc/archive.py", 207, 217)
    policy_history = tool_git_log("src/jevgc/policy.py", 1)
    error_type, error_message = _failed_read()

    return [
        ScriptedSpan(
            span=_span(
                "host-constraint",
                "agent_constraint",
                turn_index=0,
                index=0,
                input_preview="system",
                output_preview=TASK,
                output_token_count=40,
            ),
            tool="(scenario scaffolding: the host's task statement, pinned via gc.pin())",
            why="pinned -> prefilter rule 1 KEEP. Jev is never asked. used_jev=False.",
            pinned=True,
        ),
        ScriptedSpan(
            span=_span(
                "grep-principle-5",
                "grep_codebase",
                turn_index=0,
                index=1,
                input_preview=f'git grep -n -i -- "{grep_pattern}" src/jevgc',
                output_preview=grep_all,
                output_token_count=estimate_tokens(grep_all),
            ),
            tool=f'git grep -n -i -- "{grep_pattern}" src/jevgc',
            why="ambiguous -> Jev. Scored high: this is where the rule is written down.",
        ),
        ScriptedSpan(
            span=_span(
                "read-budget-allocator",
                "read_file",
                turn_index=0,
                index=2,
                input_preview="read_file src/jevgc/policy.py:148-156",
                output_preview=allocator_body,
                output_token_count=estimate_tokens(allocator_body),
            ),
            tool="read_file src/jevgc/policy.py lines 148-156",
            why="ambiguous -> Jev. Scored highest: the loop that answers the bug report.",
        ),
        ScriptedSpan(
            span=_span(
                "read-archive-budget",
                "read_file",
                turn_index=1,
                index=3,
                input_preview="read_file src/jevgc/archive.py:207-217",
                output_preview=archive_budget,
                output_token_count=estimate_tokens(archive_budget),
            ),
            tool="read_file src/jevgc/archive.py lines 207-217",
            why=(
                "dead end: a different thing called a budget (the archive's memory "
                "ceiling, not the prompt's). Scored low + DROP -> COLD."
            ),
        ),
        ScriptedSpan(
            span=_span(
                "grep-principle-5-again",
                "grep_codebase",
                turn_index=1,
                index=4,
                input_preview=f'git grep -n -i -- "{grep_pattern}" src/jevgc/policy.py src/jevgc/scorer.py',
                output_preview=grep_narrow,
                output_token_count=estimate_tokens(grep_narrow),
            ),
            tool=f'git grep -n -i -- "{grep_pattern}" src/jevgc/policy.py src/jevgc/scorer.py',
            why=(
                "duplicate-ish: a narrower re-run of the turn-0 grep, returning a "
                "subset of lines already in context. Low score + pointer -> WARM."
            ),
        ),
        ScriptedSpan(
            span=_span(
                "read-missing-module",
                "read_file",
                turn_index=1,
                index=5,
                input_preview="read_file src/jevgc/nonexistent_module.py:1-40",
                status=SpanStatus.ERROR,
                error_type=error_type,
                error_message=error_message,
                output_token_count=0,
            ),
            tool="read_file src/jevgc/nonexistent_module.py (really raised)",
            why=(
                "errored -> prefilter rule 2 KEEP, but still routed to Jev's error "
                "track: only the payload size is negotiable, never the presence."
            ),
        ),
        ScriptedSpan(
            span=_span(
                "gitlog-policy",
                "git_log",
                turn_index=2,
                index=6,
                input_preview="git log -n1 -- src/jevgc/policy.py",
                output_preview=policy_history,
                output_token_count=estimate_tokens(policy_history),
            ),
            tool="git log -n1 --pretty=format:'%h %ad %s' --date=short -- src/jevgc/policy.py",
            why="ambiguous -> Jev. Mid-high score: corroborating provenance, kept in full.",
        ),
        ScriptedSpan(
            span=_span(
                "cache-probe",
                "check_cache",
                turn_index=2,
                index=7,
                input_preview="cache_probe key=policy-allocator-148",
                output_token_count=0,
            ),
            tool="(scenario scaffolding: a zero-output cache probe)",
            why="aged out + OK + no output -> prefilter rule 5 DROP. Jev is never asked.",
        ),
    ]


def scripted_responder(requests: list[JevBatchRequest]) -> list[JevQuestionResult]:
    """Fixed Jev answers, routed by `item_id` (which is the span_id).

    An unmapped span would silently fall through to FakeJevClient's mid-confidence
    default, so this raises instead: a worked example whose answers drifted from
    what the transcript claims would be worse than no worked example.
    """
    out: list[JevQuestionResult] = []
    for req in requests:
        if req.question_id == Q_ERROR_TREATMENT:
            option, probability = JEV_ERROR_ANSWERS[req.item_id]
            out.append(
                JevQuestionResult(
                    item_id=req.item_id,
                    question_id=req.question_id,
                    choice=JevChoiceAnswer(option=option, probability=probability),
                )
            )
            continue

        score, confidence, treatment = JEV_ANSWERS[req.item_id]
        if req.question_id == Q_RELEVANCE:
            out.append(
                JevQuestionResult(
                    item_id=req.item_id,
                    question_id=req.question_id,
                    score=JevScoreAnswer(score=score, confidence=confidence),
                )
            )
        else:
            out.append(
                JevQuestionResult(
                    item_id=req.item_id,
                    question_id=req.question_id,
                    choice=JevChoiceAnswer(option=treatment, probability=confidence),
                )
            )
    return out


def example_config() -> JevGCConfig:
    """Config chosen so the scenario is actually decided rather than short-circuited.

    `keep_last_n_turns=0` with spans observed one turn after they were emitted
    is what pushes the ambiguous spans onto the Jev path; with the stock
    `keep_last_n_turns=3` an eight-span session would be kept entirely by
    recency and there would be nothing to show. `drop_zero_output_after_turns=0`
    makes the zero-output cache probe reach the prefilter's DROP rule in the
    same short session.
    """
    return JevGCConfig.from_dict(
        {
            "jev": {"api_key": "unused-fake-client"},
            "prefilter": {"keep_last_n_turns": 0, "drop_zero_output_after_turns": 0},
            "scorer": {"batch_max_size": 50, "batch_max_wait_ms": 10},
            "policy": {"relevance_keep_threshold": 0.35, "min_confidence_to_drop": 0.6},
            "telemetry": {"emit_self_metrics": False},
        }
    )


async def drive(session: list[ScriptedSpan]) -> tuple[JevGC, list[GCDecision]]:
    """Feed the session through a fresh `JevGC`, turn by turn, and collect decisions.

    Spans within a turn are gathered rather than awaited one at a time so the
    scorer batches them the way it would in a real agent loop (SPEC.md §2
    principle 6) -- awaiting serially would force one Jev round trip per span
    and misrepresent the call count the transcript reports.
    """
    jgc = JevGC(example_config(), jev_client=FakeJevClient(responder=scripted_responder))
    for scripted in session:
        if scripted.pinned:
            jgc.pin(scripted.span.span_id)

    decisions: list[GCDecision] = []
    for turn in sorted({s.span.turn_index for s in session}):
        chunk = [s.span for s in session if s.span.turn_index == turn]
        # Observe the turn's spans at the start of the *next* turn, so
        # `turns_ago == 1` and the recency short-circuit does not fire.
        jgc.advance_turn(turn + 1)
        decisions.extend(await asyncio.gather(*(jgc.observe(span) for span in chunk)))
    return jgc, decisions


def tier_of(jgc: JevGC, span_id: str) -> str:
    """Current tier of a span, via the public cold index only.

    `cold_index()` lists exactly the tracked spans that are not HOT, so absence
    from it is the public way to say "this span is HOT" without reaching into
    `JevGC._store`.
    """
    for entry in jgc.cold_index():
        if entry.span_id == span_id:
            return entry.tier.value
    return "hot"


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def _entry_dict(entry: ColdIndexEntry) -> dict[str, Any]:
    return {
        "span_id": entry.span_id,
        "tier": entry.tier.value,
        "turn_index": entry.turn_index,
        "keywords": list(entry.keywords),
    }


async def build_report() -> dict[str, Any]:
    """Run the example and return every number the text renderer prints.

    The text and `--json` outputs are two renderings of this one dict, so they
    cannot disagree -- a JSON dump that was assembled separately from the
    transcript would be a second place for the numbers to drift.
    """
    session = build_session()
    jgc, decisions = await drive(session)

    span_rows: list[dict[str, Any]] = []
    for scripted, decision in zip(session, decisions, strict=True):
        content = (
            scripted.span.output_preview
            or scripted.span.error_message
            or scripted.span.input_preview
            or ""
        )
        span_rows.append(
            {
                "turn_index": scripted.span.turn_index,
                "span_id": scripted.span.span_id,
                "name": scripted.span.name,
                "status": scripted.span.status.value,
                "tool": scripted.tool,
                "why": scripted.why,
                "content_lines": len(content.splitlines()),
                "content_chars": len(content),
                "content_preview": content.splitlines()[:2],
                "tier": decision.tier.value,
                "treatment": decision.treatment.value,
                "relevance_score": decision.relevance_score,
                "confidence": decision.confidence,
                "used_jev": decision.used_jev,
                "reason": decision.reason,
            }
        )

    context = jgc.build_context(task=TASK, budget_tokens=BUDGET_TOKENS)
    stats = jgc.stats()
    archive_stats = jgc.archive_stats()
    # Deliberate private access, as in scripts/benchmark_overhead.py: the number
    # of Jev round trips is the claim SPEC.md §2 principle 6 makes, and the
    # public facade does not expose it. This is a reporting script, not an
    # example of how to use the library.
    jev_round_trips = jgc._scorer.batches_sent

    cold_before = [_entry_dict(e) for e in jgc.cold_index()]
    hits = jgc.search_cold(SEARCH_QUERY)
    target = hits[0].span_id if hits else ""
    tier_before = tier_of(jgc, target) if target else ""
    restored = jgc.rehydrate(target) if target else None
    context_after = jgc.build_context(task=TASK, budget_tokens=BUDGET_TOKENS)

    discovery = {
        "marker": EVICTION_MARKER,
        "target_span_id": target,
        "before": {
            "marker_in_context": EVICTION_MARKER in context,
            "tier": tier_before,
            "context_tokens": estimate_tokens(context),
            "cold_index_size": len(cold_before),
        },
        "cold_index": cold_before,
        "query": SEARCH_QUERY,
        "hits": [_entry_dict(e) for e in hits],
        "restored_text": restored,
        "restored_chars": len(restored) if restored is not None else 0,
        "after": {
            "marker_in_context": EVICTION_MARKER in context_after,
            "tier": tier_of(jgc, target) if target else "",
            "context_tokens": estimate_tokens(context_after),
            "cold_index_size": len(jgc.cold_index()),
        },
    }

    # Section 3 rehydrated the evicted span on purpose, and a rehydrated span's
    # last eviction-log event is a promotion back to HOT -- the success case,
    # which `find_regret` correctly declines to call a regret. So the regret
    # pass runs against a second, identical session that was never reached into.
    replay_gc, _ = await drive(build_session())
    replay_gc.build_context(task=TASK, budget_tokens=BUDGET_TOKENS)
    findings = replay_gc.analyze_regret(
        baseline_output=BASELINE_OUTPUT, evicted_output=EVICTED_OUTPUT
    )
    control = replay_gc.analyze_regret(
        baseline_output=BASELINE_OUTPUT, evicted_output=BASELINE_OUTPUT
    )

    await jgc.aclose()
    await replay_gc.aclose()

    return {
        "meta": {
            "repo_head": repo_head(),
            "task": TASK,
            "budget_tokens": BUDGET_TOKENS,
            "jev_client": "FakeJevClient (scripted responder, no network, no API key)",
            "prefilter": {"keep_last_n_turns": 0, "drop_zero_output_after_turns": 0},
            "policy": {"relevance_keep_threshold": 0.35, "min_confidence_to_drop": 0.6},
        },
        "session": span_rows,
        "context": {
            "text": context,
            "tokens": estimate_tokens(context),
            "items": len(context.split("\n\n")) if context else 0,
            "stats": stats.model_dump(),
            "jev_round_trips": jev_round_trips,
            "archive": archive_stats.model_dump(),
        },
        "discovery": discovery,
        "regret": {
            "baseline_output": BASELINE_OUTPUT,
            "evicted_output": EVICTED_OUTPUT,
            "min_keyword_hits": 2,
            "findings": [f.model_dump(mode="json") for f in findings],
            "control_findings": len(control),
        },
    }


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------


def _rule(char: str = "=") -> str:
    return char * WIDTH


def _para(text: str, indent: str = "") -> None:
    """Word-wrap narration to WIDTH. Prose only -- never file content."""
    print(
        textwrap.fill(
            " ".join(text.split()),
            width=WIDTH,
            initial_indent=indent,
            subsequent_indent=" " * len(indent),
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _label(label: str, text: str) -> None:
    """A `label   value` row, wrapped under a hanging indent the width of the label."""
    _para(text, indent=label)


def _verbatim(text: str) -> None:
    """Print real content with every line kept inside WIDTH.

    Hard-wraps rather than reflows: this is actual file and prompt text, and a
    word-wrapper would silently reorder characters a reader may be trying to
    match against the source. A continuation is marked `~ ` so a wrap is never
    mistaken for a line break in the original.
    """
    for line in text.split("\n"):
        if not line:
            print()
            continue
        print(line[:WIDTH])
        rest = line[WIDTH:]
        while rest:
            print(f"~ {rest[: WIDTH - 2]}")
            rest = rest[WIDTH - 2 :]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


#: Column at which a `_kv` row's value starts. The wide variant is for the
#: before/after probes in section 3, whose labels are full questions.
_KV_NARROW = 32
_KV_WIDE = 54


def _kv(label: str, value: str, width: int = _KV_NARROW) -> None:
    print(f"    {label:<{width}}{value}")


def _header(report: dict[str, Any]) -> None:
    meta = report["meta"]
    print(_rule())
    print("jev-gc -- one worked example, start to finish")
    print(_rule())
    print()
    _para(
        "One multi-turn bug investigation against this repository, with every "
        "garbage-collection decision and the reason behind it."
    )
    print()
    print("REAL (measured, reproducible):")
    _label(
        "  - ",
        "tool output: actual `git grep`, actual file line-ranges and an actual "
        f"`git log` line from this repo at commit {meta['repo_head']}, plus a "
        "FileNotFoundError genuinely raised by a read that really failed",
    )
    _label("  - ", "the full pipeline: prefilter -> scorer -> policy -> store -> archive")
    _label("  - ", "the token accounting and every `reason` string shown below")
    print()
    print("SIMULATED (not evidence of anything about Jev):")
    _label(
        "  - ",
        "Jev's answers come from a FakeJevClient with a fixed scripted responder. "
        "No network, no API key. This shows that the MECHANISM works and what it "
        "costs; it says nothing about whether Jev's real judgment would score these "
        "spans this way. For that, see scripts/relevance_demo.py, which calls the "
        "live API.",
    )
    _label(
        "  - ",
        "Two of the eight spans are scaffolding rather than tool output: the pinned "
        "host constraint and a zero-output cache probe. Both are labelled below.",
    )
    print()
    print("DETERMINISM:")
    _label(
        "  - ",
        "No network, no clock reads in the span fixtures, fixed scripted answers. "
        f"Re-running at commit {meta['repo_head']} reproduces this text byte for "
        "byte. The commit sha above is the only line that tracks HEAD.",
    )
    print()
    print("FORMAT:")
    _label(
        "  - ",
        "Fixed-width plain text, no color, every line under 100 characters. Inside the "
        "two verbatim blocks below (the assembled context and the rehydrated span), a "
        "line too long for the page is hard-wrapped and its continuation is marked `~ ` "
        "-- that marker is not part of the original text.",
    )
    print()
    _label("task:    ", meta["task"])
    _label(
        "config:  ",
        "keep_last_n_turns=0  drop_zero_output_after_turns=0  "
        "relevance_keep_threshold=0.35  min_confidence_to_drop=0.6  "
        f"build_context budget={meta['budget_tokens']} tokens",
    )
    print()


def _section(number: int, title: str) -> None:
    print()
    print(_rule("-"))
    print(f"[{number}/4] {title}")
    print(_rule("-"))
    print()


def _render_session(report: dict[str, Any]) -> None:
    _section(1, "THE SESSION -- every span, every decision, every reason")
    _para(
        "Spans are observed one turn after they were emitted, so recency never "
        "short-circuits the decision. `used_jev=False` marks a span the deterministic "
        "prefilter resolved on its own, without ever asking Jev."
    )
    print()

    current_turn = -1
    for row in report["session"]:
        if row["turn_index"] != current_turn:
            current_turn = int(row["turn_index"])
            prefix = f"--- turn {current_turn} "
            print(prefix + "-" * (WIDTH - len(prefix)))
            print()

        print(f"  span_id   {row['span_id']}   (span name: {row['name']}, status {row['status']})")
        _label("  tool      ", row["tool"])
        preview = row["content_preview"] or ["(no content)"]
        for i, line in enumerate(preview):
            head = "  content   " if i == 0 else "            "
            print(head + _truncate(line, WIDTH - len(head)))
        print(f"            [{row['content_lines']} line(s), {row['content_chars']} chars]")

        score = "-" if row["relevance_score"] is None else f"{row['relevance_score']:.2f}"
        conf = "-" if row["confidence"] is None else f"{row['confidence']:.2f}"
        print(
            f"  decision  tier={row['tier'].upper():<5} "
            f"treatment={row['treatment']:<24} used_jev={row['used_jev']}"
        )
        print(f"            relevance={score:<6} confidence={conf}")
        _label("  reason    ", row["reason"])
        _label("  why       ", row["why"])
        print()

    _para(
        "Note: no span carries an item_role. JevGC leaves the scorer's optional "
        "item_role question switched off, so that field is always None -- said here "
        "rather than quietly omitted, because a missing column invites the wrong guess.",
        indent="  ",
    )
    print()


def _render_context(report: dict[str, Any]) -> None:
    ctx = report["context"]
    stats = ctx["stats"]
    _section(2, "THE ASSEMBLED CONTEXT -- what actually goes into the next prompt")
    _para(
        f"gc.build_context(task=..., budget_tokens={report['meta']['budget_tokens']}) "
        f"returned {ctx['tokens']} tokens in {ctx['items']} block(s). The errored span "
        "comes first because BudgetAllocator seats error items before it starts "
        "spending the budget:"
    )
    print()
    print(_rule("-"))
    _verbatim(ctx["text"])
    print(_rule("-"))
    print()
    print("  gc.stats() -- cumulative over all 8 spans observed, not just the blocks above:")
    _kv("spans processed", str(stats["spans_processed"]))
    _kv(
        "kept HOT / WARM / COLD",
        f"{stats['spans_kept_hot']} / {stats['spans_kept_warm']} / "
        f"{stats['spans_archived_cold']}",
    )
    _kv(
        "Jev calls / round trips",
        f"{stats['jev_calls']} span(s) scored in {ctx['jev_round_trips']} batched request(s)",
    )
    _kv("tokens rendered", str(stats["tokens_rendered"]))
    _kv("tokens at full content", f"{stats['tokens_full_content']}   (the no-GC baseline)")
    saved = int(stats["tokens_saved_estimate"])
    full = int(stats["tokens_full_content"])
    pct = saved / full * 100 if full else 0.0
    _kv("tokens_saved_estimate", f"{saved}   ({pct:.1f}% of the baseline)")
    _kv(
        "archive retained",
        f"{ctx['archive']['retained_bytes']} bytes in {ctx['archive']['entry_count']} "
        f"entries, {ctx['archive']['released_count']} released",
    )
    print()
    _para(
        "Read those two token numbers carefully, because they answer different "
        "questions. `tokens_rendered` is the sum of all 8 spans as jev-gc stored them, "
        "decision by decision. It is NOT the size of the block printed above -- that is "
        f"the budget-limited subset actually selected for this one prompt, "
        f"{ctx['tokens']} tokens. `tokens_full_content` is what those same 8 spans would "
        "have cost verbatim with no GC at all. The gap between those two is the claim; "
        "everything above it is the audit trail for the claim.",
        indent="  ",
    )
    print()


def _render_discovery(report: dict[str, Any]) -> None:
    d = report["discovery"]
    _section(3, "DISCOVERY AND REHYDRATION -- an eviction that is reversible")
    _para(
        "Several turns on, the investigation has moved: the agent now wants to know why "
        "a rehydrate can come back without content. It half-remembers reading something "
        "about exactly that -- but the span was evicted, so it cannot name the span_id."
    )
    print()
    print("  BEFORE")
    _kv(f"{d['marker']!r} present in the assembled context?", str(d["before"]["marker_in_context"]), _KV_WIDE)
    _kv(f"tier of {d['target_span_id']}", d["before"]["tier"], _KV_WIDE)
    _kv("assembled context", f"{d['before']['context_tokens']} tokens", _KV_WIDE)
    _kv("cold index", f"{d['before']['cold_index_size']} entries", _KV_WIDE)
    print()
    _para(
        "The cold index is all the agent can still see of the evicted spans -- keywords "
        "only, cheap enough to sit in every prompt. `turn=` is the turn jev-gc observed "
        "the span, one after the turn that emitted it.",
        indent="  ",
    )
    print()
    for entry in d["cold_index"]:
        print(f"    {entry['span_id']:<24} tier={entry['tier']:<5} turn={entry['turn_index']}")
        _label("      keywords: ", ", ".join(entry["keywords"]))
    print()
    _para(f"gc.search_cold({d['query']!r})", indent="  ")
    print()
    if not d["hits"]:
        print("    (no hits)")
    for i, entry in enumerate(d["hits"], start=1):
        print(
            f"    {i}. span_id={entry['span_id']}  tier={entry['tier']}  "
            f"turn={entry['turn_index']}"
        )
        _label("       keywords: ", ", ".join(entry["keywords"]))
    print()
    _para(
        f"gc.rehydrate({d['target_span_id']!r}) -> {d['restored_chars']} chars, restored "
        "verbatim from the snapshot the archive took when the span was first observed:",
        indent="  ",
    )
    print()
    print(_rule("-"))
    _verbatim(d["restored_text"] or "(nothing restored)")
    print(_rule("-"))
    print()
    print("  AFTER")
    _kv(f"{d['marker']!r} present in the assembled context?", str(d["after"]["marker_in_context"]), _KV_WIDE)
    _kv(f"tier of {d['target_span_id']}", d["after"]["tier"], _KV_WIDE)
    delta = int(d["after"]["context_tokens"]) - int(d["before"]["context_tokens"])
    _kv("assembled context", f"{d['after']['context_tokens']} tokens ({delta:+d})", _KV_WIDE)
    _kv("cold index", f"{d['after']['cold_index_size']} entries", _KV_WIDE)
    print()
    _para(
        "What makes this more than a tier flip: the restored text is an immutable "
        "snapshot taken at observation time, not a fresh call to the tool. The agent "
        "gets back exactly what jev-gc evicted -- which is also what any audit of that "
        "eviction decision has to read.",
        indent="  ",
    )
    print()


def _render_regret(report: dict[str, Any]) -> None:
    r = report["regret"]
    _section(4, "REGRET ANALYSIS -- holding the eviction to account")
    _para(
        "The host runs the same task twice, once with full context and once with jev-gc "
        "evicting, then hands both final answers to gc.analyze_regret(). Content that "
        "surfaces in the full-context answer but not in the GC'd one is evidence that an "
        "eviction cost the agent something."
    )
    print()
    _para(
        "Run against a second, identical session rather than the one above: section 3 "
        "deliberately rehydrated the evicted span, and a rehydrated span's last "
        "eviction-log event is a promotion back to HOT -- the success case, which "
        "find_regret correctly declines to call a regret.",
        indent="  ",
    )
    print()
    print("  baseline_output (full-context run):")
    _para(r["baseline_output"], indent="    ")
    print()
    print("  evicted_output (jev-gc run):")
    _para(r["evicted_output"], indent="    ")
    print()
    _para(
        f"gc.analyze_regret(min_keyword_hits={r['min_keyword_hits']}) -> "
        f"{len(r['findings'])} finding(s)",
        indent="  ",
    )
    print()
    for i, f in enumerate(r["findings"], start=1):
        print(
            f"    {i}. span_id={f['span_id']}  evicted_to={f['evicted_to']}  "
            f"turn={f['turn_index']}  regret_score={f['regret_score']:.2f}"
        )
        _label("       eviction reason:  ", f["eviction_reason"])
        _label("       matched keywords: ", ", ".join(f["matched_keywords"]))
    print()
    _para(
        f"Control -- the same session with the two answers made identical: "
        f"{r['control_findings']} finding(s). A method that flagged evictions regardless "
        "of the outputs would be worthless, so the null case is worth printing too.",
        indent="  ",
    )
    print()
    _para(
        "What regret_score actually is: the fraction of the evicted span's archived "
        "keywords that appear in the full-context answer and are absent from the GC'd "
        "one. Keyword overlap, nothing more. It is correlational -- two runs can diverge "
        "for reasons unrelated to the eviction -- and it under-reports, because an "
        "eviction can degrade reasoning without changing which nouns come out. Use it to "
        "rank evictions worth inspecting and to tune relevance_keep_threshold, never as "
        "proof that an eviction was safe.",
        indent="  ",
    )
    print()


def render_text(report: dict[str, Any]) -> None:
    _header(report)
    _render_session(report)
    _render_context(report)
    _render_discovery(report)
    _render_regret(report)
    stats = report["context"]["stats"]
    print(_rule())
    _para(
        f"Bottom line for this session: 8 spans that would have cost "
        f"{stats['tokens_full_content']} tokens verbatim were stored as "
        f"{stats['tokens_rendered']}, the next prompt drew "
        f"{report['context']['tokens']} of those from a "
        f"{report['meta']['budget_tokens']}-token budget, the errored span survived "
        "compression, nothing was irrecoverable, and the one eviction that cost an "
        "answer could be named afterwards. The Jev answers driving those decisions were "
        "scripted; the payloads, the pipeline and the arithmetic were not."
    )
    print(_rule())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the same data as JSON instead of the transcript",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = asyncio.run(build_report())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    render_text(report)


if __name__ == "__main__":
    main()
