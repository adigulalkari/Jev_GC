from __future__ import annotations

import asyncio

import pytest

from jevgc.exceptions import JevTimeoutError
from jevgc.jev_client.fakes import FakeJevClient
from jevgc.models import JevChoiceAnswer, JevQuestionResult, JevScoreAnswer
from jevgc.scorer import JevBatchScorer


@pytest.mark.asyncio
async def test_batches_on_size(make_span):
    client = FakeJevClient()
    scorer = JevBatchScorer(client, batch_max_size=2, batch_max_wait_ms=10_000)

    span_a, span_b = make_span(span_id="a"), make_span(span_id="b")
    results = await asyncio.gather(
        scorer.submit(span_a, item_id="a"),
        scorer.submit(span_b, item_id="b"),
    )

    assert len(client.calls) == 1  # one batch covering both submissions
    assert {r.item_id for r in results} == {"a", "b"}
    assert all(r.used_jev for r in results)


@pytest.mark.asyncio
async def test_batches_on_timeout_without_reaching_max_size(make_span):
    # a tiny real wait (not a mocked clock) keeps this fast (<30s suite budget)
    # while still exercising the actual timer path, not just the size path.
    client = FakeJevClient()
    scorer = JevBatchScorer(client, batch_max_size=50, batch_max_wait_ms=10)

    span = make_span()
    result = await asyncio.wait_for(scorer.submit(span, item_id="only"), timeout=2.0)

    assert len(client.calls) == 1
    assert result.item_id == "only"
    assert result.used_jev is True


@pytest.mark.asyncio
async def test_fails_open_on_client_error(make_span):
    client = FakeJevClient(raises=JevTimeoutError("timed out"))
    scorer = JevBatchScorer(client, batch_max_size=1, batch_max_wait_ms=10_000)

    span = make_span()
    result = await scorer.submit(span, item_id="x")

    assert result.used_jev is False
    assert result.score is None
    assert result.choice is None
    assert scorer.batch_error_count == 1


@pytest.mark.asyncio
async def test_error_span_only_asks_error_treatment(make_span):
    from jevgc.models import SpanStatus

    seen_question_ids: list[str] = []

    def responder(requests):
        seen_question_ids.extend(req.question_id for req in requests)
        return [
            JevQuestionResult(
                item_id=req.item_id,
                question_id=req.question_id,
                choice=JevChoiceAnswer(option="keep_error_summary_only", probability=0.8),
            )
            for req in requests
        ]

    client = FakeJevClient(responder=responder)
    scorer = JevBatchScorer(client, batch_max_size=1, batch_max_wait_ms=10_000)

    span = make_span(status=SpanStatus.ERROR, error_type="ValueError")
    result = await scorer.submit(span, item_id="e1")

    assert seen_question_ids == ["error_treatment"]
    assert result.score is None
    assert result.choice is not None
    assert result.choice.option == "keep_error_summary_only"


@pytest.mark.asyncio
async def test_relevance_and_treatment_merged_into_one_result(make_span):
    def responder(requests):
        out = []
        for req in requests:
            if req.question_id == "relevance":
                out.append(
                    JevQuestionResult(
                        item_id=req.item_id, question_id=req.question_id,
                        score=JevScoreAnswer(score=0.8, confidence=0.9),
                    )
                )
            else:
                out.append(
                    JevQuestionResult(
                        item_id=req.item_id, question_id=req.question_id,
                        choice=JevChoiceAnswer(option="include_full", probability=0.7),
                    )
                )
        return out

    client = FakeJevClient(responder=responder)
    scorer = JevBatchScorer(client, batch_max_size=1, batch_max_wait_ms=10_000)

    span = make_span()
    result = await scorer.submit(span, item_id="s1")

    assert result.score is not None and result.score.score == 0.8
    assert result.choice is not None and result.choice.option == "include_full"
