"""Jev API client.

This is the one file in the codebase that is allowed to know the concrete
Jev HTTP contract (SPEC.md §5 flags it as provisional and isolates it here
deliberately, via the `JevClient` Protocol). The contract implemented below
was verified live against `docs.typesafe.ai` and `api.typesafe.ai` and
differs from SPEC.md's sketch in a few ways worth recording:

- Endpoint is `POST {base_url}/systemone` (base_url already includes
  `/v1`), not `/v1/score`.
- Auth is `Authorization: Bearer <key>`.
- The request's `questions` field is a *dict* keyed by question id, not a
  list -- each question is `{"type": "choice"|"score"|"noul",
  "instructions": str, "criteria": ...}`. `choice` criteria is a
  `dict[option, description]`; `score` criteria is an ordered
  `list[str]` rubric (a "legend"); there is no free continuous score
  parameter to hand the API -- score answers come back as an index into
  that legend, not a raw 0.0-1.0 float.
- The response's `answers` field is also a dict keyed by question id, with
  per-type payloads (`choice`+`probabilities`+`confidence`,
  `score`+`legend`+`probabilities`+`confidence`, `noul`).

Two consequences for jev-gc's design, contained entirely to this file:

1. `relevance_to_current_task` (SPEC.md §4.4 Q1) is asked as a `score`
   question against a fixed 5-point relevance rubric, and the returned
   legend index is normalized to `[0.0, 1.0]` by dividing by
   `len(criteria) - 1` before being wrapped in `JevQuestionResult` --
   this is what lets the rest of the codebase keep treating relevance as a
   continuous score per `models.JevScoreAnswer`, matching the frozen
   contract.
2. Jev's public API only accepts one `state` string per call, but
   SPEC.md's batching design (§5) needs N spans' worth of distinct
   content scored in a single HTTP round trip. `HTTPJevClient` resolves
   this by rendering each request's `state` under a
   `=== ITEM <item_id> ===` heading inside one composite state document,
   and prefixing every question id with `"{item_id}::"` (exactly the
   scheme SPEC.md already specifies for splitting the response back out).
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from jevgc.exceptions import JevAPIError, JevRateLimitError, JevTimeoutError
from jevgc.models import JevChoiceAnswer, JevQuestionResult, JevScoreAnswer

logger = logging.getLogger(__name__)

QuestionType = Literal["choice", "score"]

#: Fixed relevance rubric used for every `relevance_to_current_task` Score
#: question. Ordered low -> high; the returned legend index is normalized
#: against `len(RELEVANCE_RUBRIC) - 1`.
RELEVANCE_RUBRIC: list[str] = [
    "Irrelevant to the current task; no bearing on it",
    "Tangentially related background, unlikely to be needed",
    "Plausibly useful if the task circles back to this",
    "Directly supports reasoning about the current task",
    "Critical fact the current task depends on",
]


class JevBatchRequest(BaseModel):
    """One question about one item, as handed to `JevClient.batch_ask`.

    `state` is the item-scoped content (task description + span content)
    this question should be evaluated against -- see module docstring for
    how multiple requests' `state`s are composed into one API call.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    question_id: str
    type: QuestionType
    instructions: str
    state: str
    # `choice` only: option -> human-readable description.
    options: dict[str, str] | None = None
    # `score` only: ordered rubric; defaults to the relevance rubric.
    criteria: list[str] | None = None


class JevClient(Protocol):
    """Structurally-typed so `FakeJevClient` needs no inheritance."""

    async def batch_ask(self, requests: list[JevBatchRequest]) -> list[JevQuestionResult]: ...


class HTTPJevClient:
    """Real Jev API client, over `httpx.AsyncClient`."""

    def __init__(
        self,
        api_key: SecretStr,
        base_url: str = "https://api.typesafe.ai/v1",
        timeout_seconds: float = 2.0,
        max_retries: int = 2,
        model: str = "jev-latest",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._model = model
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout_seconds)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def batch_ask(self, requests: list[JevBatchRequest]) -> list[JevQuestionResult]:
        if not requests:
            return []

        state, questions, id_map = _compose_request(requests)
        payload = {"state": state, "model": self._model, "questions": questions}

        retrying = retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential_jitter(initial=0.1, max=2.0),
            retry=retry_if_exception_type((JevRateLimitError, httpx.TransportError)),
        )
        response_body = await retrying(self._post)(payload)
        return _parse_response(response_body, id_map)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._http.post(
                f"{self._base_url}/systemone",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.TimeoutException as exc:
            raise JevTimeoutError(f"Jev API timed out after {self._timeout_seconds}s") from exc
        except httpx.TransportError:
            raise

        if response.status_code == 429:
            raise JevRateLimitError("Jev API rate limit exceeded", status_code=429)
        if response.status_code >= 400:
            raise JevAPIError(
                f"Jev API returned {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
            )

        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise JevAPIError("Jev API returned a malformed (non-JSON) body") from exc

        if "answers" not in body:
            raise JevAPIError("Jev API response missing 'answers' field")

        return body


def _compose_request(
    requests: list[JevBatchRequest],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, tuple[str, str]]]:
    """Build the composite `state` text, the `questions` dict keyed by
    `"{item_id}::{question_id}"`, and a map back from that composite key to
    `(item_id, question_id)` for splitting the response."""
    states_by_item: dict[str, str] = {}
    for req in requests:
        states_by_item.setdefault(req.item_id, req.state)

    state_sections = [
        f"=== ITEM {item_id} ===\n{item_state}" for item_id, item_state in states_by_item.items()
    ]
    composite_state = "\n\n".join(state_sections)

    questions: dict[str, dict[str, Any]] = {}
    id_map: dict[str, tuple[str, str]] = {}
    for req in requests:
        composite_key = f"{req.item_id}::{req.question_id}"
        id_map[composite_key] = (req.item_id, req.question_id)

        scoped_instructions = f"For ITEM {req.item_id}: {req.instructions}"
        if req.type == "choice":
            questions[composite_key] = {
                "type": "choice",
                "instructions": scoped_instructions,
                "criteria": req.options or {},
            }
        else:
            questions[composite_key] = {
                "type": "score",
                "instructions": scoped_instructions,
                "criteria": req.criteria or RELEVANCE_RUBRIC,
            }

    return composite_state, questions, id_map


def _parse_response(
    body: dict[str, Any], id_map: dict[str, tuple[str, str]]
) -> list[JevQuestionResult]:
    answers = body.get("answers", {})
    results: list[JevQuestionResult] = []

    for composite_key, (item_id, question_id) in id_map.items():
        answer = answers.get(composite_key)
        if answer is None:
            logger.warning("Jev response missing answer for %s", composite_key)
            continue

        answer_type = answer.get("type")
        choice_answer: JevChoiceAnswer | None = None
        score_answer: JevScoreAnswer | None = None

        try:
            if answer_type == "choice":
                option = answer["choice"]
                probabilities = answer.get("probabilities", {})
                probability = float(probabilities.get(option, answer.get("confidence", 1.0)))
                choice_answer = JevChoiceAnswer(option=option, probability=probability)
            elif answer_type == "score":
                legend = answer.get("legend", {})
                max_index = max((int(k) for k in legend), default=len(RELEVANCE_RUBRIC) - 1)
                raw_score = float(answer["score"])
                normalized = raw_score / max_index if max_index > 0 else raw_score
                score_answer = JevScoreAnswer(
                    score=min(max(normalized, 0.0), 1.0),
                    confidence=float(answer.get("confidence", 0.0)),
                )
            else:
                logger.warning("Unrecognized Jev answer type %r for %s", answer_type, composite_key)
                continue
        except (KeyError, TypeError, ValueError) as exc:
            raise JevAPIError(f"Malformed answer for {composite_key}: {exc}") from exc

        results.append(
            JevQuestionResult(
                item_id=item_id,
                question_id=question_id,
                choice=choice_answer,
                score=score_answer,
            )
        )

    return results
