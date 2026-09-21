"""Exception hierarchy for jev-gc.

Design rule: a Jev API failure of any kind must never crash the host agent's
loop. Every public entry point in `gc.py` catches `JevGCError` and its
subclasses internally and degrades to the fail-open policy (see policy.py).
These types exist for logging, metrics, and for callers who explicitly want
to observe failures (e.g. in tests), not as control flow the host agent is
expected to handle itself.
"""

from __future__ import annotations


class JevGCError(Exception):
    """Base class for all jev-gc errors."""


class ConfigurationError(JevGCError):
    """Raised at startup for invalid/missing configuration. This is the one
    exception type that IS allowed to propagate and stop the process --
    fail fast on misconfiguration, fail open on runtime API issues."""


class JevAPIError(JevGCError):
    """Non-timeout, non-rate-limit failure from the Jev API (4xx/5xx,
    malformed response, schema mismatch)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class JevRateLimitError(JevAPIError):
    """429 from the Jev API. Callers should back off; the batch scorer
    retries with jittered backoff before giving up and failing open."""


class JevTimeoutError(JevGCError):
    """The Jev API did not respond within the configured timeout."""


class StoreError(JevGCError):
    """Failure in a context-store backend (in-memory, SQLite, Redis, ...)."""


class OTelIntegrationError(JevGCError):
    """Raised when the OTel span processor cannot translate a ReadableSpan
    into a SpanRecord (unexpected shape) -- logged and the span is skipped,
    never raised into the host application's export pipeline."""
