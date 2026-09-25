# Quickstart

## Install

```bash
pip install jev-gc
```

Optional extras: `pip install "jev-gc[langgraph]"` or `"jev-gc[strands]"`.

To run the test suite or the bundled examples from a clone instead:

```bash
git clone https://github.com/adigulalkari/Jev_GC
cd Jev_GC
pip install -e ".[dev,langgraph,strands]"
```

## Configure

Copy the example config and point it at your Jev API key:

```bash
cp config/jevgc.example.yaml jevgc.yaml
export JEV_API_KEY=...   # from the TypeSafe dashboard
```

## Wire it to an OTel-instrumented agent

```python
from opentelemetry.sdk.trace import TracerProvider
from jevgc import JevGC

provider = TracerProvider()
gc = JevGC.from_config("jevgc.yaml")
gc.attach_to_tracer_provider(provider)

# ... agent runs, spans stream in via `provider` ...

# after each turn, or whenever the host is about to call the LLM again:
context = gc.build_context(task="Reconcile invoice #4521", budget_tokens=8000)
```

## No OTel instrumentation yet? Use the manual escape hatch

```python
from jevgc.models import SpanRecord, SpanStatus

span = SpanRecord(
    span_id="tool-call-1",
    trace_id="trace-1",
    name="query_erp_table",
    status=SpanStatus.OK,
    start_time_unix_ns=...,
    end_time_unix_ns=...,
    output_preview="invoice #4521 total: $4521.00",
    output_token_count=12,
    turn_index=0,
)
decision = await gc.observe(span)
```

## Running without a real Jev key

Every test and the Strands example both run against `FakeJevClient` -- no
network, no key required:

```python
from jevgc import JevGC, JevGCConfig
from jevgc.jev_client.fakes import FakeJevClient

config = JevGCConfig.from_dict({"jev": {"api_key": "unused"}})
gc = JevGC(config, jev_client=FakeJevClient())
```

## Try the full example

```bash
export JEV_API_KEY=...
export MISTRAL_API_KEY=...
pip install -e ".[examples]"
python examples/codebase_triage_agent/agent.py
```

See [`examples/codebase_triage_agent/README.md`](../examples/codebase_triage_agent/README.md).
