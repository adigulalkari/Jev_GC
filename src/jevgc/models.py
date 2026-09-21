"""Core data models shared across jev-gc.

These are the load-bearing contracts: the OTel processor, the pre-filter, the Jev
scorer, the policy/budget allocator, and the tiered store all speak these types.
Changing a field here is a breaking change for every other module -- treat this
file as frozen once implementation of the other modules begins.
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpanStatus(str, enum.Enum):
    """Mirrors OpenTelemetry's StatusCode, decoupled from the SDK enum so this
    module has zero hard dependency on `opentelemetry-api` internals leaking
    through the public model surface."""

    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


class Tier(str, enum.Enum):
    """Where a context item currently lives.

    HOT   -- included verbatim in the next prompt.
    WARM  -- evicted from the prompt, replaced by a one-line pointer; full
             content still retrievable by span_id without calling Jev again.
    COLD  -- archived to long-term storage (e.g. a vector store); only
             reachable via explicit retrieval, never auto-reinstated.
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class Treatment(str, enum.Enum):
    """Jev's Choice-question answer for a successful (non-error) span."""

    INCLUDE_FULL = "include_full"
    INCLUDE_SUMMARY_ONLY = "include_summary_only"
    KEEP_POINTER_ONLY = "keep_pointer_only"
    DROP = "drop"


class ErrorTreatment(str, enum.Enum):
    """Jev's Choice-question answer for an errored span. Errored spans are
    never dropped outright -- only their payload size is negotiable."""

    KEEP_FULL_TRACE = "keep_full_trace"
    KEEP_ERROR_SUMMARY_ONLY = "keep_error_summary_only"
    KEEP_ERROR_TYPE_ONLY = "keep_error_type_only"


class ItemRole(str, enum.Enum):
    """Structural classification of a span relative to the current task.
    Used for prioritization when the token budget is tight, and for
    dashboards/debugging -- not a decision field on its own."""

    CRITICAL_FACT = "critical_fact"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    DEAD_END = "dead_end"
    DUPLICATE_OF_EXISTING = "duplicate_of_existing"
    UNCERTAIN = "uncertain"


class SpanRecord(BaseModel):
    """Normalized view of one OTel span, as handed to the pre-filter.

    This is intentionally decoupled from `opentelemetry.sdk.trace.ReadableSpan`:
    the otel/processor module is responsible for translating a real span into
    this shape. Every other module only ever sees a `SpanRecord`.
    """

    model_config = ConfigDict(frozen=True)

    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    name: str
    status: SpanStatus
    start_time_unix_ns: int
    end_time_unix_ns: int
    attributes: dict[str, Any] = Field(default_factory=dict)
    # GenAI semantic-convention fields, extracted for convenience (may be None
    # if the span doesn't carry them -- e.g. a plain DB call span).
    gen_ai_operation_name: str | None = None
    gen_ai_tool_name: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    input_preview: str | None = None
    output_preview: str | None = None
    output_token_count: int | None = None
    turn_index: int = 0

    @property
    def duration_ms(self) -> float:
        return (self.end_time_unix_ns - self.start_time_unix_ns) / 1_000_000

    @property
    def is_error(self) -> bool:
        return self.status == SpanStatus.ERROR


class JevChoiceAnswer(BaseModel):
    """Raw shape of a single Choice-question answer from the Jev API."""

    model_config = ConfigDict(frozen=True)

    option: str
    probability: float = Field(ge=0.0, le=1.0)


class JevScoreAnswer(BaseModel):
    """Raw shape of a single Score-question answer from the Jev API."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class JevQuestionResult(BaseModel):
    """One question's result within a batched Jev response, keyed back to the
    span it was asked about via `item_id`."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    question_id: str
    choice: JevChoiceAnswer | None = None
    score: JevScoreAnswer | None = None


class GCDecision(BaseModel):
    """The final, code-owned verdict for one span. Jev never produces this
    type directly -- only the policy module (policy.py) does, by combining
    Jev's raw answers with deterministic rules (recency, pin status,
    confidence thresholds, budget)."""

    model_config = ConfigDict(frozen=True)

    span_id: str
    tier: Tier
    treatment: Treatment | ErrorTreatment
    relevance_score: float | None = None
    confidence: float | None = None
    role: ItemRole | None = None
    reason: str  # human-readable, always populated, for debugging/audit
    used_jev: bool  # False when the decision was made by the deterministic
    # pre-filter alone and Jev was never called for this span


class ContextItem(BaseModel):
    """What actually gets assembled into the next prompt: either the
    span's full content, a compressed summary, or a pointer."""

    model_config = ConfigDict(frozen=True)

    span_id: str
    tier: Tier
    rendered_text: str
    token_count: int
    decision: GCDecision


def new_item_id() -> str:
    """Stable, collision-resistant id used to correlate a span with its Jev
    answer inside a batched request/response pair."""
    return uuid.uuid4().hex


def now_unix_ns() -> int:
    return time.time_ns()
