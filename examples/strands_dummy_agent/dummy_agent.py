"""A small, runnable Strands agent that emits realistic OpenTelemetry spans
for jev-gc to consume (SPEC.md §9).

The agent itself is not the point -- it's a Gemini-backed research
assistant with three genuinely useful tools (a live weather lookup, a safe
calculator, and a local docs search) that Strands' native OTel
instrumentation turns into tool-call spans automatically. One of the
prompts below deliberately triggers a tool error (an unknown city) and one
deliberately triggers an irrelevant tool call (a calculation unrelated to
the actual question), so the demo has real material for jev-gc's
error-handling and relevance-scoring paths.

Run:
    export JEV_API_KEY=...      # optional -- falls back to FakeJevClient
    export GEMINI_API_KEY=...   # required -- this example needs a real LLM
    python examples/strands_dummy_agent/dummy_agent.py

Uses Gemini's `gemini-3.6-flash` model (fast + free-tier friendly) and
keeps the whole run to a handful of turns since the free tier is rate
limited -- this is a demo, not a load test.
"""

from __future__ import annotations

import ast
import asyncio
import operator
import os
import re
from pathlib import Path

import httpx
from strands import Agent, tool
from strands.models.gemini import GeminiModel
from strands.telemetry import StrandsTelemetry

from jevgc.config import JevGCConfig
from jevgc.gc import JevGC
from jevgc.jev_client.fakes import FakeJevClient

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Strands' model-provider retry logic can silently swallow a fast, explicit
#: API error (e.g. a 429) and keep retrying well past any reasonable wait --
#: observed hanging indefinitely against Gemini's free tier rather than
#: surfacing the underlying error. Bounding each turn here turns that into a
#: clear, actionable timeout instead of an indefinite hang.
_TURN_TIMEOUT_S = 45

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


@tool
def get_weather(city: str) -> str:
    """Look up the current temperature for a city using the free Open-Meteo API.

    Args:
        city: City name, e.g. "Tokyo" or "San Francisco".
    """
    geocode = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10.0,
    ).json()
    results = geocode.get("results")
    if not results:
        raise ValueError(f"No location found for city {city!r}")

    lat, lon = results[0]["latitude"], results[0]["longitude"]
    forecast = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current": "temperature_2m"},
        timeout=10.0,
    ).json()
    temp = forecast["current"]["temperature_2m"]
    return f"{results[0]['name']}: {temp}°C right now"


@tool
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression (+, -, *, /, **). No names, no calls.

    Args:
        expression: e.g. "12 * (3 + 4)".
    """
    return _safe_eval(ast.parse(expression, mode="eval").body)


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


@tool
def search_docs(query: str) -> str:
    """Search this repo's README and docs/ for lines mentioning `query`.

    Args:
        query: Search term, case-insensitive.
    """
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    hits: list[str] = []
    for path in [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.name}:{lineno}: {line.strip()}")
    if not hits:
        return f"No matches for {query!r} in README.md or docs/."
    return "\n".join(hits[:10])


def build_agent() -> Agent:
    model = GeminiModel(
        client_args={"api_key": os.environ["GEMINI_API_KEY"]},
        model_id="gemini-3.6-flash",
    )
    return Agent(
        model=model,
        tools=[get_weather, calculate, search_docs],
        system_prompt=(
            "You are a concise research assistant. Use tools when they help answer "
            "the question. Keep replies to 2-3 sentences."
        ),
    )


def build_jevgc() -> JevGC:
    api_key = os.environ.get("JEV_API_KEY")
    config = JevGCConfig.from_dict(
        {
            "jev": {"api_key": api_key or "unused-fake-key"},
            "policy": {"relevance_keep_threshold": 0.35, "min_confidence_to_drop": 0.6},
        }
    )
    # No JEV_API_KEY set -> run entirely offline against FakeJevClient, per
    # SPEC.md §9: this example must work without a real Jev key.
    jev_client = None if api_key else FakeJevClient()
    return JevGC(config, jev_client=jev_client)


PROMPTS = [
    "What's the weather in Tokyo right now?",
    "Quick unrelated aside: what's 17 * 24?",
    "What does this repo's README say jev-gc does, in one line?",
    "What's the weather in Notarealcityxyz123?",
]


async def main() -> None:
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("Set GEMINI_API_KEY to run this example (see module docstring).")

    # StrandsTelemetry(), called with no tracer_provider, both creates a
    # TracerProvider *and* registers it as the OTel global -- Strands' own
    # Agent binds its tracer to whatever the global provider is at agent-
    # construction time, so this has to happen before build_agent() below.
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
                "check your GEMINI_API_KEY's quota at https://ai.dev/rate-limit"
            )
        except Exception as exc:  # noqa: BLE001 - demo script, keep going on tool/model errors
            print(f"agent turn raised (continuing): {exc}")

    await gc.wait_all()

    context = gc.build_context(task="Summarize what we learned about Tokyo's weather", budget_tokens=800)
    print("\n=== assembled context (budget=800 tokens) ===")
    print(context)

    stats = gc.stats()
    print("\n=== jev-gc stats ===")
    print(stats.model_dump_json(indent=2))

    await gc.aclose()


if __name__ == "__main__":
    asyncio.run(main())
