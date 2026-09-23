"""upstream.py: SSE parsing, error classification, Retry-After, transport errors."""

from __future__ import annotations

import json

import httpx
import pytest

from app.fallback import FAILOVER, FATAL, RETRY, UpstreamError
from app.providers import Provider
from app.upstream import (
    _is_retryable_status,
    open_chat_stream,
    parse_retry_after,
    parse_sse_content,
    parse_sse_event,
    post_chat,
    post_embeddings,
    stream_chat,
    streaming_body,
)

PROVIDER = Provider(name="up", base_url="https://up.test/v1", api_key_env="UPSTREAM_TEST_KEY")


@pytest.fixture(autouse=True)
def provider_key(monkeypatch):
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "k")


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_parse_sse_helpers():
    line = 'data: {"choices":[{"delta":{"content":"Hi"}}],"usage":{"total_tokens":3}}'
    assert parse_sse_content(line) == ("Hi", {"total_tokens": 3})
    assert parse_sse_content("data: [DONE]") == ("", None)
    assert parse_sse_content(": keep-alive comment") == ("", None)
    assert parse_sse_content("data: {not json") == ("", None)
    assert parse_sse_event("data: [1, 2]") is None
    assert parse_sse_event("data:") is None
    assert parse_sse_event('data: {"a": 1}') == {"a": 1}


def test_streaming_body_forces_usage_but_keeps_caller_options():
    body = streaming_body({"model": "m", "stream_options": {"foo": 1}})
    assert body["stream"] is True
    assert body["stream_options"] == {"foo": 1, "include_usage": True}
    assert streaming_body({"stream_options": None})["stream_options"] == {"include_usage": True}


def test_retry_after_parsing():
    assert parse_retry_after(httpx.Headers({"retry-after": "3"})) == 3.0
    assert parse_retry_after(httpx.Headers({"retry-after-ms": "1500", "retry-after": "9"})) == 1.5
    assert parse_retry_after(httpx.Headers({"retry-after-ms": "soon", "retry-after": "2"})) == 2.0
    assert parse_retry_after(httpx.Headers({"retry-after": "-4"})) == 0.0
    date = parse_retry_after(httpx.Headers({"retry-after": "Wed, 21 Oct 2015 07:28:10 GMT"}), now=1445412480.0)
    assert date == 10.0
    assert parse_retry_after(httpx.Headers({"retry-after": "whenever"})) is None
    assert parse_retry_after(httpx.Headers({})) is None


def test_legacy_retryable_helper():
    assert _is_retryable_status(503) and _is_retryable_status(429)
    assert not _is_retryable_status(401) and not _is_retryable_status(400)


@pytest.mark.parametrize("status,kind", [(500, RETRY), (429, RETRY), (401, FAILOVER), (404, FAILOVER), (422, FATAL)])
async def test_http_errors_are_classified(status, kind):
    async with client_for(lambda r: httpx.Response(status, text="nope", headers={"retry-after": "1"})) as client:
        with pytest.raises(UpstreamError) as exc:
            await post_chat(client, PROVIDER, {"model": "m", "messages": []})
    assert exc.value.status_code == status and exc.value.kind == kind
    assert exc.value.retry_after == 1.0
    assert exc.value.message == "up: nope"


async def test_success_sends_auth_and_forces_non_streaming():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    async with client_for(handler) as client:
        assert await post_chat(client, PROVIDER, {"model": "m", "stream": True}) == {"ok": True}
        assert await post_embeddings(client, PROVIDER, {"model": "e", "input": "x"}) == {"ok": True}
    assert seen["auth"] == "Bearer k"
    assert seen["body"] == {"model": "e", "input": "x"}


async def test_bad_upstream_bodies_are_retryable():
    async with client_for(lambda r: httpx.Response(200, text="<html>oops</html>")) as client:
        with pytest.raises(UpstreamError) as exc:
            await post_chat(client, PROVIDER, {"model": "m"})
    assert exc.value.kind == RETRY and "not JSON" in exc.value.message
    async with client_for(lambda r: httpx.Response(200, json=[1, 2])) as client:
        with pytest.raises(UpstreamError, match="unexpected response shape"):
            await post_embeddings(client, PROVIDER, {"model": "e"})


async def test_missing_key_is_a_failover_without_a_request(monkeypatch):
    monkeypatch.delenv("UPSTREAM_TEST_KEY")
    calls = []
    async with client_for(lambda r: calls.append(r) or httpx.Response(200, json={})) as client:
        for call in (post_chat(client, PROVIDER, {}), post_embeddings(client, PROVIDER, {}),
                     open_chat_stream(client, PROVIDER, {})):
            with pytest.raises(UpstreamError) as exc:
                await call
            assert exc.value.kind == FAILOVER and exc.value.reason == "no-key"
    assert calls == []


@pytest.mark.parametrize("error,reason", [
    (httpx.ConnectTimeout("slow"), "timeout"),
    (httpx.ConnectError("refused"), "connect-error"),
    (httpx.RemoteProtocolError("garbage"), "transport-error"),
])
async def test_transport_errors_are_retryable(error, reason):
    def handler(request):
        raise error

    async with client_for(handler) as client:
        with pytest.raises(UpstreamError) as exc:
            await post_embeddings(client, PROVIDER, {})
        assert exc.value.kind == RETRY and exc.value.reason == reason and exc.value.status_code == 504
        with pytest.raises(UpstreamError) as exc:
            await open_chat_stream(client, PROVIDER, {})
        assert exc.value.reason == reason


async def test_stream_chat_yields_complete_events():
    sse = 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n: ping\n\ndata: [DONE]\n\n'
    async with client_for(lambda r: httpx.Response(200, text=sse)) as client:
        events = [e async for e in stream_chat(client, PROVIDER, {"model": "m"})]
    assert events == [b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n', b": ping\n\n", b"data: [DONE]\n\n"]


async def test_stream_error_status_is_raised_before_any_byte():
    async with client_for(lambda r: httpx.Response(503, text="overloaded")) as client:
        with pytest.raises(UpstreamError) as exc:
            async for _ in stream_chat(client, PROVIDER, {"model": "m"}):
                pytest.fail("no event expected")
    assert exc.value.status_code == 503 and exc.value.message == "up: overloaded"


class DroppingStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[]}\n\n'
        raise httpx.ReadError("reset")


async def test_stream_chat_reports_mid_stream_drops():
    async with client_for(lambda r: httpx.Response(200, stream=DroppingStream())) as client:
        got = []
        with pytest.raises(UpstreamError) as exc:
            async for event in stream_chat(client, PROVIDER, {"model": "m"}):
                got.append(event)
    assert got == [b'data: {"choices":[]}\n\n']
    assert exc.value.reason == "transport-error"


class FailingErrorBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        raise httpx.ReadError("reset while reading the error body")
        yield b""  # pragma: no cover


async def test_unreadable_error_body_still_classifies_the_status():
    async with client_for(lambda r: httpx.Response(500, stream=FailingErrorBody())) as client:
        with pytest.raises(UpstreamError) as exc:
            await open_chat_stream(client, PROVIDER, {"model": "m"})
    assert exc.value.status_code == 500 and exc.value.message == "up: "
