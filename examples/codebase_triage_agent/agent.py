"""A Mistral-backed bug-triage agent that investigates real bug reports
against *this actual repository* -- jev-gc's own source, tests, and git
history. Unlike the weather-demo example, every tool call here operates
on real, non-trivial content (source files, git log, pytest output), so a
session naturally produces the things jev-gc exists to manage: irrelevant
grep hits, dead-end file reads, and real tool failures, across enough
turns that the deterministic pre-filter alone can't resolve everything --
this is where Jev's relevance/treatment judgment actually earns its keep
(contrast with examples/strands_dummy_agent/, where a 4-turn demo barely
touches Jev at all).

Tools are real subprocess/filesystem calls against this repo -- no
mocking, no fabricated data:

  grep_codebase  -- `git grep` for a pattern under a path
  read_file      -- read a line range of a real file (raises if missing)
  git_log        -- real commit history for a path
  run_pytest     -- actually runs a test file and reports pass/fail

Run:
    export JEV_API_KEY=...       # optional -- falls back to FakeJevClient
    export MISTRAL_API_KEY=...   # required -- this example needs a real LLM
    python examples/codebase_triage_agent/agent.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from strands import Agent, tool
from strands.models.mistral import MistralModel
from strands.telemetry import StrandsTelemetry

from jevgc.config import JevGCConfig
from jevgc.gc import JevGC
from jevgc.jev_client.fakes import FakeJevClient

REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBPROCESS_TIMEOUT_S = 30
#: Strands' model-provider retry logic can silently swallow a fast, explicit
#: API error (e.g. a 429) and keep retrying well past any reasonable wait --
#: observed hanging indefinitely against both Gemini and Mistral free tiers
#: rather than surfacing the underlying error. Bounding each turn here turns
#: that into a clear, actionable timeout instead of an indefinite hang.
_TURN_TIMEOUT_S = 45


def _run(args: list[str], cwd: Path = REPO_ROOT) -> tuple[int, str]:
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S
    )
    output = result.stdout + result.stderr
    return result.returncode, output


@tool
def grep_codebase(pattern: str, path: str = ".") -> str:
    """Search this repo's tracked files for a literal pattern using `git grep`.

    Args:
        pattern: Text to search for (case-insensitive, not a regex).
        path: Directory or file to search under, relative to the repo root.
    """
    returncode, output = _run(["git", "grep", "-n", "-i", "--", pattern, path])
    if returncode == 1:  # git grep: no matches found
        return f"No matches for {pattern!r} under {path!r}."
    if returncode != 0:
        raise RuntimeError(f"git grep failed (exit {returncode}): {output.strip()}")
    lines = output.strip().splitlines()
    return "\n".join(lines[:30]) + (f"\n... ({len(lines) - 30} more)" if len(lines) > 30 else "")


@tool
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Read a range of lines from a file in this repo, relative to the repo root.

    Args:
        path: File path relative to the repo root, e.g. "src/jevgc/policy.py".
        start_line: First line to include (1-indexed).
        end_line: Last line to include (inclusive); defaults to start_line + 60.
    """
    target = (REPO_ROOT / path).resolve()
    if REPO_ROOT not in target.parents and target != REPO_ROOT:
        raise ValueError(f"{path!r} escapes the repo root")
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {path!r}")

    lines = target.read_text().splitlines()
    end = end_line or (start_line + 60)
    excerpt = lines[max(start_line - 1, 0) : end]
    numbered = [f"{i}: {line}" for i, line in enumerate(excerpt, start=start_line)]
    return "\n".join(numbered)


@tool
def git_log(path: str, max_commits: int = 5) -> str:
    """Show recent commit history touching a path in this repo.

    Args:
        path: File or directory path relative to the repo root.
        max_commits: Maximum number of commits to return.
    """
    returncode, output = _run(
        ["git", "log", f"-n{max_commits}", "--pretty=format:%h %ad %s", "--date=short", "--", path]
    )
    if returncode != 0:
        raise RuntimeError(f"git log failed (exit {returncode}): {output.strip()}")
    return output.strip() or f"No commit history for {path!r}."


@tool
def run_pytest(test_path: str) -> str:
    """Run a specific test file or test node id with pytest and report the result.

    Args:
        test_path: e.g. "tests/unit/test_policy.py" or "tests/unit/test_policy.py::test_foo".
    """
    returncode, output = _run([sys.executable, "-m", "pytest", "-q", test_path])
    status = "PASSED" if returncode == 0 else "FAILED"
    tail = "\n".join(output.strip().splitlines()[-15:])
    return f"{status} (exit {returncode})\n{tail}"


def build_agent() -> Agent:
    model = MistralModel(
        api_key=os.environ["MISTRAL_API_KEY"],
        model_id="mistral-small-latest",
    )
    return Agent(
        model=model,
        tools=[grep_codebase, read_file, git_log, run_pytest],
        system_prompt=(
            "You are a bug-triage assistant investigating reports against a real Python "
            "codebase (jev-gc). Use the tools to actually look at the code, tests, and git "
            "history before answering -- don't guess. Keep your final answer to 3-4 sentences "
            "citing what you found (file/line or commit)."
        ),
    )


def build_jevgc() -> JevGC:
    api_key = os.environ.get("JEV_API_KEY")
    config = JevGCConfig.from_dict(
        {
            "jev": {"api_key": api_key or "unused-fake-key"},
            # Tighter than the weather demo's defaults: this session runs long
            # enough (many tool calls across 4 investigations) that recency
            # alone would keep almost everything, defeating the point of the
            # comparison -- see README's "does this actually matter" section.
            "prefilter": {"keep_last_n_turns": 1, "drop_zero_output_after_turns": 20},
            "policy": {"relevance_keep_threshold": 0.35, "min_confidence_to_drop": 0.6},
        }
    )
    jev_client = None if api_key else FakeJevClient()
    return JevGC(config, jev_client=jev_client)


PROMPTS = [
    "A bug report claims build_context() can silently drop ALL errored spans if their "
    "combined token count exceeds the budget. Investigate BudgetAllocator in policy.py and "
    "its tests -- is this actually possible?",
    "The Strands example agent's model id was retired at some point (a 404 NOT_FOUND error "
    "from Gemini). Find the commit that fixed it and summarize what changed.",
    "Run the test suite for context_builder.py and tell me if it's passing.",
    "Read src/jevgc/nonexistent_module.py and describe what's in it.",
]


async def main() -> None:
    if "MISTRAL_API_KEY" not in os.environ:
        raise SystemExit("Set MISTRAL_API_KEY to run this example (see module docstring).")

    telemetry = StrandsTelemetry()
    gc = build_jevgc()
    gc.attach_to_tracer_provider(telemetry.tracer_provider)

    agent = build_agent()

    for turn, prompt in enumerate(PROMPTS):
        print(f"\n=== turn {turn}: {prompt}")
        gc.advance_turn(turn)
        try:
            response = await asyncio.wait_for(agent.invoke_async(prompt), timeout=_TURN_TIMEOUT_S)
            print(f"agent: {response}")
        except asyncio.TimeoutError:  # noqa: UP041 -- asyncio.TimeoutError, not builtins.TimeoutError, for py3.10 compat
            print(
                f"agent turn timed out after {_TURN_TIMEOUT_S}s (continuing) -- this usually means "
                "the model provider is stuck retrying a rate-limit error rather than raising it; "
                "check your MISTRAL_API_KEY's plan/billing status at console.mistral.ai"
            )
        except Exception as exc:  # noqa: BLE001 - demo script, keep going on tool/model errors
            print(f"agent turn raised (continuing): {exc}")

    await gc.wait_all()

    context = gc.build_context(
        task="Summarize the BudgetAllocator error-handling investigation", budget_tokens=1200
    )
    print("\n=== assembled context (budget=1200 tokens) ===")
    print(context)

    stats = gc.stats()
    print("\n=== jev-gc stats ===")
    print(stats.model_dump_json(indent=2))

    await gc.aclose()


if __name__ == "__main__":
    asyncio.run(main())
