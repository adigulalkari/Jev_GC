from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import SecretStr

from jevgc.exceptions import JevAPIError, JevRateLimitError, JevTimeoutError
from jevgc.jev_client.client import HTTPJevClient, JevBatchRequest

BASE_URL = "https://api.typesafe.ai/v1"


def _client() -> HTTPJevClient:
    return HTTPJevClient(api_key=SecretStr("test-key"), base_url=BASE_URL, max_retries=2)


def _request(item_id: str = "item1", qid: str = "relevance") -> JevBatchRequest:
    return JevBatchRequest(
        item_id=item_id,
        question_id=qid,
        type="score",
        instructions="how relevant?",
        state="some span content",
    )


@pytest.mark.asyncio
@respx.mock
async def test_happy_path_score():
    respx.post(f"{BASE_URL}/systemone").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "item1::relevance": {
                        "type": "score",
                        "score": 3.0,
                        "confidence": 0.9,
                        "legend": {"0": "a", "1": "b", "2": "c", "3": "d", "4": "e"},
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )
    )
    client = _client()
    results = await client.batch_ask([_request()])

    assert len(results) == 1
    assert results[0].score is not None
    assert results[0].score.score == pytest.approx(3.0 / 4)
    assert results[0].score.confidence == 0.9


@pytest.mark.asyncio
@respx.mock
async def test_happy_path_choice():
    respx.post(f"{BASE_URL}/systemone").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "item1::treatment": {
                        "type": "choice",
                        "choice": "include_full",
                        "confidence": 0.6,
                        "probabilities": {"include_full": 0.75},
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )
    )
    client = _client()
    req = JevBatchRequest(
        item_id="item1",
        question_id="treatment",
        type="choice",
        instructions="how to treat?",
        state="content",
        options={"include_full": "keep it all"},
    )
    results = await client.batch_ask([req])

    assert results[0].choice is not None
    assert results[0].choice.option == "include_full"
    assert results[0].choice.probability == 0.75


@pytest.mark.asyncio
@respx.mock
async def test_429_retries_then_succeeds():
    route = respx.post(f"{BASE_URL}/systemone")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(
            200,
            json={
                "answers": {
                    "item1::relevance": {"type": "score", "score": 2.0, "confidence": 0.5, "legend": {}}
                }
            },
        ),
    ]
    client = _client()
    results = await client.batch_ask([_request()])
    assert route.call_count == 2
    assert results[0].score is not None


@pytest.mark.asyncio
@respx.mock
async def test_429_retry_exhausted_raises_rate_limit_error():
    respx.post(f"{BASE_URL}/systemone").mock(return_value=httpx.Response(429))
    client = HTTPJevClient(api_key=SecretStr("k"), base_url=BASE_URL, max_retries=1)

    with pytest.raises(JevRateLimitError):
        await client.batch_ask([_request()])


@pytest.mark.asyncio
@respx.mock
async def test_malformed_body_raises_api_error():
    respx.post(f"{BASE_URL}/systemone").mock(
        return_value=httpx.Response(200, content=b"not json")
    )
    client = _client()
    with pytest.raises(JevAPIError):
        await client.batch_ask([_request()])


@pytest.mark.asyncio
@respx.mock
async def test_timeout_raises_jev_timeout_error():
    respx.post(f"{BASE_URL}/systemone").mock(side_effect=httpx.TimeoutException("boom"))
    client = _client()
    with pytest.raises(JevTimeoutError):
        await client.batch_ask([_request()])


@pytest.mark.asyncio
async def test_empty_requests_short_circuits_without_http_call():
    client = _client()
    assert await client.batch_ask([]) == []
