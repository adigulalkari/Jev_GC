"""FakeJevClient: no-network stand-in for `JevClient`, used by every unit
test that touches the scorer or policy, and by the Strands example so it
runs without a real Jev API key."""

from __future__ import annotations

from collections.abc import Callable

from jevgc.jev_client.client import JevBatchRequest
from jevgc.models import JevChoiceAnswer, JevQuestionResult, JevScoreAnswer

ResponseFn = Callable[[list[JevBatchRequest]], list[JevQuestionResult]]


class FakeJevClient:
    """Structurally satisfies `JevClient`.

    Configure with either a canned `responses` map (`(item_id, question_id)
    -> JevQuestionResult`) or a `responder` callable for dynamic behavior.
    If neither answers a given request, defaults to a mid-confidence
    "include_summary_only" / 0.5 relevance answer so tests that don't care
    about exact values still get something structurally valid.
    """

    def __init__(
        self,
        responses: dict[tuple[str, str], JevQuestionResult] | None = None,
        responder: ResponseFn | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._responses = responses or {}
        self._responder = responder
        self._raises = raises
        self.calls: list[list[JevBatchRequest]] = []

    async def batch_ask(self, requests: list[JevBatchRequest]) -> list[JevQuestionResult]:
        self.calls.append(requests)

        if self._raises is not None:
            raise self._raises

        if self._responder is not None:
            return self._responder(requests)

        results: list[JevQuestionResult] = []
        for req in requests:
            canned = self._responses.get((req.item_id, req.question_id))
            if canned is not None:
                results.append(canned)
                continue

            if req.type == "choice":
                default_option = next(iter(req.options or {"include_summary_only": ""}))
                results.append(
                    JevQuestionResult(
                        item_id=req.item_id,
                        question_id=req.question_id,
                        choice=JevChoiceAnswer(option=default_option, probability=0.5),
                    )
                )
            else:
                results.append(
                    JevQuestionResult(
                        item_id=req.item_id,
                        question_id=req.question_id,
                        score=JevScoreAnswer(score=0.5, confidence=0.5),
                    )
                )
        return results
