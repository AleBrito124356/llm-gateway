"""Failover, retries and the circuit breaker, driven through the HTTP API."""

from __future__ import annotations

import time

import httpx

from .conftest import auth, chat, sse_events


class BrokenStream(httpx.AsyncByteStream):
    """Sends one SSE chunk, then the connection drops."""

    async def __aiter__(self):
        yield (
            b'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
            b'"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        )
        raise httpx.ReadError("connection reset by peer")

    async def aclose(self) -> None:
        return None


def stream(client, content="stream it", model="big", **extra):
    return client.post("/v1/chat/completions", headers=auth(), json={
        "model": model, "stream": True, "messages": [{"role": "user", "content": content}], **extra,
    })


def test_revoked_provider_key_fails_over_across_providers(gw, fake):
    fake.fail_always("meta/big", 401)
    fake.fail_always("meta/small", 403)
    r = chat(gw, "who serves me?", model="big")
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "local"
    assert r.headers["x-upstream-model"] == "llama"
    assert r.headers["x-gateway-attempts"] == "nim:meta/big=401, nim:meta/small=403, local:llama=200"
    assert r.json()["choices"][0]["message"]["content"] == "llama says: who serves me?"


def test_retired_model_404_fails_over_without_retrying(gw, fake):
    fake.fail_always("meta/big", 404)
    r = chat(gw, "hello there", model="big")
    assert r.status_code == 200
    assert [c["body"]["model"] for c in fake.chat_calls()] == ["meta/big", "meta/small"]


def test_missing_provider_key_fails_over_instantly(make_gateway, fake):
    # Huge backoff: any sleep at all would make this test take 10+ seconds.
    gw = make_gateway(backoff_base_seconds=10, backoff_cap_seconds=10, max_retries_per_target=2)
    started = time.perf_counter()
    r = chat(gw, "no key here", model="keyless-chat")
    elapsed = time.perf_counter() - started
    assert r.status_code == 200
    assert elapsed < 1.0
    assert r.headers["x-gateway-attempts"] == "keyless:whatever=no-key, local:llama=200"
    assert all(c["host"] != "keyless.test" for c in fake.calls)


def test_transient_503_is_retried_on_the_same_target(gw, fake):
    fake.script("meta/big", 503)
    r = chat(gw, "try again", model="big")
    assert r.status_code == 200
    assert r.headers["x-gateway-attempts"] == "nim:meta/big=503, nim:meta/big=200"


def test_upstream_retry_after_is_honoured(make_gateway, fake):
    gw = make_gateway(backoff_cap_seconds=5)
    fake.script("meta/big", (429, {"retry-after": "0"}))
    r = chat(gw, "rate limited upstream", model="big")
    assert r.status_code == 200
    assert r.headers["x-gateway-attempts"] == "nim:meta/big=429, nim:meta/big=200"


def test_upstream_retry_after_beyond_cap_fails_over_now(make_gateway, fake):
    gw = make_gateway(backoff_cap_seconds=5, max_retries_per_target=3)
    fake.script("meta/big", (429, {"retry-after": "120"}))
    started = time.perf_counter()
    r = chat(gw, "busy upstream", model="big")
    assert time.perf_counter() - started < 1.0
    assert r.headers["x-gateway-attempts"] == "nim:meta/big=429, nim:meta/small=200"


def test_caller_error_is_returned_without_failover(gw, fake):
    fake.script("meta/big", 400)
    r = chat(gw, "malformed for upstream", model="big")
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "upstream_error"
    assert len(fake.chat_calls()) == 1
    assert r.headers["x-gateway-attempts"] == "nim:meta/big=400"


def test_exhausted_chain_is_a_502_listing_every_hop(gw, fake):
    for model in ("meta/big", "meta/small", "llama"):
        fake.fail_always(model, 500)
    r = chat(gw, "nobody home", model="big")
    assert r.status_code == 502
    assert r.headers["x-gateway-attempts"] == (
        "nim:meta/big=500, nim:meta/big=500, nim:meta/small=500, nim:meta/small=500, local:llama=500, local:llama=500"
    )
    assert "all targets failed" in r.json()["error"]["message"]


def test_breaker_skips_a_dead_primary_and_reports_it(make_gateway, fake):
    gw = make_gateway(breaker_failure_threshold=2, breaker_cooldown_seconds=60)
    fake.fail_always("meta/big", 503)
    first = chat(gw, "one", model="big")
    assert first.status_code == 200
    assert first.headers["x-gateway-attempts"] == "nim:meta/big=503, nim:meta/big=503, nim:meta/small=200"
    calls_to_big = sum(1 for c in fake.chat_calls() if c["body"]["model"] == "meta/big")

    second = chat(gw, "two", model="big")
    assert second.status_code == 200
    assert second.headers["x-gateway-attempts"] == "nim:meta/big=circuit-open, nim:meta/small=200"
    assert sum(1 for c in fake.chat_calls() if c["body"]["model"] == "meta/big") == calls_to_big

    health = gw.get("/admin/health").json()
    assert health["status"] == "degraded"
    assert health["unhealthy_targets"] == ["nim:meta/big"]
    big = health["targets"]["nim:meta/big"]
    assert big["state"] == "open" and big["last_status"] == 503 and big["retry_in_seconds"] > 50
    assert health["targets"]["nim:meta/small"]["successes"] == 2
    assert health["targets"]["local:llama"]["state"] == "closed"  # configured but never called
    assert health["providers"]["keyless"]["api_key_set"] is False
    assert "nim-secret" not in str(health)


def test_every_circuit_open_is_a_fast_503(make_gateway, fake):
    gw = make_gateway(routing={"fallbacks": {}}, breaker_failure_threshold=1, breaker_cooldown_seconds=60)
    fake.fail_always("meta/big", 503)
    assert chat(gw, "first", model="big").status_code == 502
    r = chat(gw, "second", model="big")
    assert r.status_code == 503
    assert r.headers["x-gateway-attempts"] == "nim:meta/big=circuit-open"
    assert 55 <= int(r.headers["retry-after"]) <= 60


def test_stream_fails_over_before_the_first_byte(gw, fake):
    fake.script("meta/big", 503, 503)
    r = stream(gw, "stream via fallback")
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "nim"
    assert r.headers["x-upstream-model"] == "meta/small"
    assert r.headers["x-gateway-attempts"] == "nim:meta/big=503, nim:meta/big=503, nim:meta/small=200"
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in sse_events(r) if isinstance(e, dict))
    assert text == "meta/small says: stream via fallback"


def test_stream_with_no_healthy_target_is_an_http_error_not_an_sse_200(gw, fake):
    for model in ("meta/big", "meta/small", "llama"):
        fake.fail_always(model, 503)
    r = stream(gw, "doomed stream")
    assert r.status_code == 502
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["error"]["type"] == "upstream_error"


def test_stream_interrupted_midway_ends_with_an_error_event(gw, fake):
    fake.script("meta/small", httpx.Response(200, headers={"content-type": "text/event-stream"},
                                             stream=BrokenStream()))
    r = stream(gw, "cut me off", model="small")
    assert r.status_code == 200
    events = sse_events(r)
    assert events[0]["choices"][0]["delta"]["content"] == "partial"
    assert "stream interrupted" in events[1]["error"]["message"]
    assert events[-1] == "[DONE]"
    # The partial answer was billed but not cached.
    usage = gw.get("/admin/usage?format=json").json()
    assert usage["totals"]["requests"] == 1
    assert chat(gw, "cut me off", model="small").headers["x-cache"] == "MISS"
    health = gw.get("/admin/health").json()
    assert health["targets"]["nim:meta/small"]["failures"] == 1


def test_embeddings_fail_over_and_name_the_model_that_answered(gw, fake):
    fake.fail_always("embedder", 404)
    r = gw.post("/v1/embeddings", headers=auth(), json={"model": "embed", "input": "vectors please"})
    assert r.status_code == 200
    assert r.headers["x-gateway-model"] == "local-embed"
    assert r.headers["x-gateway-attempts"] == "nim:embedder=404, local:nomic=200"


def test_embeddings_exhausted_chain_is_a_502(gw, fake):
    fake.fail_always("embedder", 500)
    fake.fail_always("nomic", 500)
    r = gw.post("/v1/embeddings", headers=auth(), json={"model": "embed", "input": "x"})
    assert r.status_code == 502
    assert r.headers["x-gateway-model"] == "embed"


def test_transport_errors_are_retried_then_failed_over(make_gateway, fake):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "nim.test":
            raise httpx.ConnectError("connection refused")
        return fake(request)

    gw = make_gateway(transport=httpx.MockTransport(handler))
    r = chat(gw, "is anyone there", model="big")
    assert r.status_code == 200
    assert r.headers["x-gateway-attempts"] == (
        "nim:meta/big=connect-error, nim:meta/big=connect-error, "
        "nim:meta/small=connect-error, nim:meta/small=connect-error, local:llama=200"
    )
