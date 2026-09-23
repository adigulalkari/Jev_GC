"""jev-gc: real-time, OpenTelemetry-driven context garbage collection for LLM agents.

jev-gc watches the OpenTelemetry spans an agent already emits (tool calls, retriever
calls, LLM calls) and decides, in real time, what stays in the next prompt's context
window and what gets evicted to cheap storage. Cheap, high-volume relevance/role
judgments are delegated to Jev (TypeSafe AI's "System One" decision model); anything
that doesn't fit a clean classification is never silently dropped -- it fails open.

Typical usage:

    from jevgc import JevGC, JevGCConfig

    gc = JevGC.from_config("jevgc.yaml")
    gc.attach_to_tracer_provider(trace.get_tracer_provider())

    # ... agent runs, spans stream in ...

    context = gc.build_context(task="...", budget_tokens=8000)
"""

from jevgc.archive import ArchiveEntry, ColdIndexEntry, EvictionEvent
from jevgc.config import JevGCConfig
from jevgc.exceptions import (
    ConfigurationError,
    JevAPIError,
    JevGCError,
    JevRateLimitError,
    JevTimeoutError,
)
from jevgc.gc import JevGC
from jevgc.models import (
    ContextItem,
    ErrorTreatment,
    GCDecision,
    SpanRecord,
    Tier,
    Treatment,
)
from jevgc.regret import RegretFinding, find_regret

__version__ = "0.1.0"

__all__ = [
    "JevGC",
    "JevGCConfig",
    "SpanRecord",
    "ContextItem",
    "GCDecision",
    "Tier",
    "Treatment",
    "ErrorTreatment",
    "ArchiveEntry",
    "ColdIndexEntry",
    "EvictionEvent",
    "RegretFinding",
    "find_regret",
    "JevGCError",
    "JevAPIError",
    "JevRateLimitError",
    "JevTimeoutError",
    "ConfigurationError",
]
