# Security Policy

## Supported versions

jev-gc is pre-1.0. Only the latest release on `main` receives security fixes.

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1 | No |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it through GitHub's private vulnerability reporting:
[Report a vulnerability](https://github.com/adigulalkari/Jev_GC/security/advisories/new).

Include what you did, what happened, and what you expected. A minimal
reproduction helps more than anything else. Expect an initial response within a
week; this is a small project and there is no on-call rotation.

## What is in scope

jev-gc sits between an agent's telemetry and its prompts, which makes a few
classes of bug security-relevant rather than merely wrong:

- **API key leakage.** `jev.api_key` is a `pydantic.SecretStr` and must never
  appear in a log line, an exception message, a `repr()`, or a span attribute.
  A path that leaks it is a vulnerability, not a bug. There is a test asserting
  this; a way around it is worth reporting.
- **Span content leakage.** Span payloads routinely contain whatever the
  agent's tools returned, which can include customer data. Anything that writes
  span content somewhere unexpected — logs at default levels, error messages,
  telemetry attributes — is in scope.
- **Content crossing session boundaries.** The archive and the cold index are
  per-`JevGC` instance. Anything that lets content from one instance surface in
  another is in scope.
- **Denial of service through unbounded growth.** The archive is bounded by
  `archive.max_content_bytes`; the eviction log is not. An input that makes
  memory grow without limit is worth reporting.

## What is out of scope

- The fact that evicted content is retained in memory by design. That is the
  documented purpose of the archive, and it is bounded — see `archive.py`.
- Jev returning a wrong relevance judgment. Misclassification is expected and
  is the reason every Jev call has a code-owned fail-open fallback; a bad score
  is a tuning question, not a vulnerability.
- Anything requiring an attacker who already controls the host process.

## Handling secrets in reports

When attaching logs or reproductions, redact API keys and any real span content
first. If a report would require sending us sensitive payloads, describe the
shape of the data instead and we will work from that.
