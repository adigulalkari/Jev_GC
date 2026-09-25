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

Get a Jev API key from the TypeSafe dashboard:

```bash
export JEV_API_KEY=...
```

`JevGCConfig` only requires `jev.api_key` -- everything else has a default, so
the fastest path needs no file at all:

```python
import os
from jevgc.config import JevGCConfig

config = JevGCConfig.from_dict({"jev": {"api_key": os.environ["JEV_API_KEY"]}})
```

Prefer a YAML file? `config/jevgc.example.yaml` documents every field and
ships in this repo -- clone it, or copy its contents into a `jevgc.yaml` of
your own (that file is not part of the PyPI package, so `pip install jev-gc`
alone won't put it on disk for you):

```bash
git clone https://github.com/adigulalkari/Jev_GC   # only needed for this file
cp Jev_GC/config/jevgc.example.yaml jevgc.yaml
```

```python
from jevgc.config import JevGCConfig

config = JevGCConfig.from_yaml("jevgc.yaml")   # interpolates ${JEV_API_KEY} from the env
```

## Wire it to an OTel-instrumented agent

```python
from opentelemetry.sdk.trace import TracerProvider
from jevgc import JevGC

provider = TracerProvider()
gc = JevGC(config)
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
