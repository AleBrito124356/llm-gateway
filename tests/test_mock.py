"""The offline mock provider, its transports, and the local/hash embedder."""

from __future__ import annotations

import base64
import json

import httpx
import numpy as np
import pytest

from app.mock import (
    GatewayTransport,
    MockUpstream,
    OfflineGuardTransport,
    build_mocks,
    count_tokens,
    hash_embedding,
)
from app.providers import ConfigError, Provider, ProviderRegistry

from .conftest import PROVIDERS, auth, chat


def cos(a: str, b: str) -> float:
    return float(np.dot(hash_embedding(a), hash_embedding(b)))


def test_hash_embedding_is_deterministic_and_normalised():
    v = hash_embedding("semantic caching saves money")
    assert v == hash_embedding("semantic caching saves money")
    assert len(v) == 384
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert len(hash_embedding("x y z", dims=64)) == 64
    assert np.linalg.norm(hash_embedding("")) == 0.0


def test_hash_embedding_ranks_paraphrases_above_other_topics():
    assert cos("What is the capital of France?", "Tell me the capital of France, please.") > 0.99
    assert cos("How do I reverse a list in Python?", "How can I reverse a Python list?") > 0.85
    assert cos("What is the capital of France?", "What is the capital of Spain?") < 0.6
    assert cos("Write a haiku about caching.", "Explain TCP congestion control.") < 0.2


def test_count_tokens_is_roughly_bpe_sized():
    assert count_tokens("") == 0
    assert count_tokens("hi you") == 2
    assert count_tokens("hi there") == 3
    assert count_tokens("internationalization") == 5


def mock(**options) -> MockUpstream:
    return MockUpstream(Provider(name="m", base_url="http://m.mock.invalid/v1", type="mock", options=options))


async def call(upstream: MockUpstream, path: str, body=None, method: str = "POST") -> httpx.Response:
    request = httpx.Request(method, f"http://m.mock.invalid/v1{path}", json=body)
    response = await upstream.handle(request)
    await response.aread()
    return response


async def test_chat_completion_shape_and_usage():
    r = await call(mock(), "/chat/completions", {"model": "llama", "messages": [
        {"role": "system", "content": "be brief"}, {"role": "user", "content": "What is 2 + 2?"}]})
    data = r.json()
    assert r.status_code == 200
    assert data["object"] == "chat.completion" and data["model"] == "llama"
    assert 'you asked "What is 2 + 2?"' in data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"
    usage = data["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"] > 10


async def test_chat_honours_max_tokens_json_mode_tools_and_n():
    m = mock()
    short = (await call(m, "/chat/completions", {"model": "x", "max_tokens": 3,
                                                 "messages": [{"role": "user", "content": "hello"}]})).json()
    assert short["choices"][0]["finish_reason"] == "length"
    assert short["usage"]["completion_tokens"] <= 3
    js = (await call(m, "/chat/completions", {"model": "x", "response_format": {"type": "json_object"},
                                              "messages": [{"role": "user", "content": "colors"}]})).json()
    assert json.loads(js["choices"][0]["message"]["content"])["echo"] == "colors"
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    tc = (await call(m, "/chat/completions", {"model": "x", "tools": tools,
                                              "messages": [{"role": "user", "content": "weather?"}]})).json()
    choice = tc["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    none = (await call(m, "/chat/completions", {"model": "x", "tools": tools, "tool_choice": "none",
                                                "messages": [{"role": "user", "content": "weather?"}]})).json()
    assert none["choices"][0]["message"]["content"]
    two = (await call(m, "/chat/completions", {"model": "x", "n": 2,
                                               "messages": [{"role": "user", "content": "hi"}]})).json()
    assert [c["index"] for c in two["choices"]] == [0, 1]


async def test_streaming_sends_usage_chunk_only_when_asked():
    m = mock()
    body = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "stream a sentence please"}]}
    plain = await call(m, "/chat/completions", body)
    events = [line[6:] for line in plain.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["choices"] for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert 'you asked "stream a sentence please"' in text
    with_usage = await call(m, "/chat/completions", {**body, "stream_options": {"include_usage": True}})
    last = json.loads([line[6:] for line in with_usage.text.splitlines() if line.startswith("data: ")][-2])
    assert last["choices"] == [] and last["usage"]["completion_tokens"] > 0


async def test_embeddings_float_and_base64():
    m = mock(embedding_dims=16)
    floats = (await call(m, "/embeddings", {"model": "e", "input": ["a b", "c d"]})).json()
    assert [len(d["embedding"]) for d in floats["data"]] == [16, 16]
    b64 = (await call(m, "/embeddings", {"model": "e", "input": "a b", "encoding_format": "base64"})).json()
    decoded = np.frombuffer(base64.b64decode(b64["data"][0]["embedding"]), dtype="<f4")
    assert np.allclose(decoded, floats["data"][0]["embedding"], atol=1e-6)
    tokens = (await call(m, "/embeddings", {"model": "e", "input": [1, 2, 3], "dimensions": 8})).json()
    assert len(tokens["data"][0]["embedding"]) == 8
    assert (await call(m, "/embeddings", {"model": "e"})).status_code == 400


async def test_failure_injection_first_n_then_recover():
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    m = MockUpstream(Provider(name="m", base_url="http://m.mock.invalid/v1", type="mock",
                              options={"fail_first_n": 2, "retry_after": 1, "latency_ms": 250}), sleep=fake_sleep)
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    first = await call(m, "/chat/completions", body)
    assert first.status_code == 503 and first.headers["retry-after"] == "1"
    assert (await call(m, "/chat/completions", body)).status_code == 503
    assert (await call(m, "/chat/completions", body)).status_code == 200
    assert sleeps == [0.25, 0.25, 0.25]
    assert len(m.calls) == 3


async def test_bad_requests_and_other_paths():
    m = mock()
    assert (await call(m, "/chat/completions", {"model": "x", "messages": []})).status_code == 400
    assert (await call(m, "/nope", {"a": 1})).status_code == 404
    assert (await call(m, "/models", method="GET")).json() == {"object": "list", "data": []}
    assert (await call(m, "/chat/completions", method="DELETE")).status_code == 405
    raw = await m.handle(httpx.Request("POST", "http://m.mock.invalid/v1/chat/completions", content=b"{not json"))
    assert raw.status_code == 400


def test_mock_provider_config_validation():
    reg = ProviderRegistry.from_dict({"providers": {"Local Mock": {"type": "mock", "fail_status": 503}}, "models": {}})
    provider = reg.providers["Local Mock"]
    assert provider.base_url == "http://local-mock.mock.invalid/v1"
    assert provider.options == {"fail_status": 503}
    assert provider.api_key() == "mock"
    for bad, message in [
        ({"p": {"type": "grpc", "base_url": "x"}}, "unknown type"),
        ({"p": {"base_url": "http://x", "fail_status": 503}}, "only applies"),
        ({"p": {"type": "mock", "fail_status": 200}}, "between"),
        ({"p": {"type": "mock", "fail_first_n": "lots"}}, "must be a number"),
        ({"p": {"type": "mock", "latency_ms": True}}, "between"),
        ({"a": {"type": "mock", "base_url": "http://same/v1"}, "b": {"type": "mock", "base_url": "http://same/v1"}},
         "share the host"),
        ({"p": {"base_url": "http://x"}, "q": "not a mapping"}, "must be a mapping"),
    ]:
        with pytest.raises(ConfigError, match=message):
            ProviderRegistry.from_dict({"providers": bad, "models": {}})


async def test_gateway_transport_routes_mock_hosts_in_process():
    registry = ProviderRegistry.from_dict({
        "providers": {"fake": {"type": "mock"}, "real": {"base_url": "https://real.example/v1"}},
        "models": {},
    })
    guard = OfflineGuardTransport()
    transport = GatewayTransport(guard, build_mocks(registry))
    async with httpx.AsyncClient(transport=transport) as client:
        ok = await client.post("http://fake.mock.invalid/v1/embeddings", json={"model": "e", "input": "hi"})
        assert ok.status_code == 200
        with pytest.raises(httpx.ConnectError, match="offline mode"):
            await client.post("https://real.example/v1/chat/completions", json={})
    assert guard.blocked == ["https://real.example/v1/chat/completions"]
    assert transport.mock_for("fake") is not None and transport.mock_for("real") is None


def test_mock_providers_serve_the_gateway_end_to_end(make_gateway, fake):
    providers = {
        "providers": {**PROVIDERS["providers"], "mockp": {"type": "mock"}},
        "models": {**PROVIDERS["models"], "mocked": {"provider": "mockp", "upstream_model": "mm"}},
    }
    gw = make_gateway(providers=providers)
    r = chat(gw, "hello mock", model="mocked")
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "mockp"
    assert r.json()["model"] == "mm"
    assert not fake.chat_calls()  # never reached the injected transport
    mock_upstream = gw.app.state.transport.mock_for("mockp")
    assert mock_upstream.calls[-1]["body"]["model"] == "mm"


def test_local_hash_embedder_needs_no_provider(make_gateway, fake):
    gw = make_gateway(embed_model="local/hash", cache_similarity_threshold=0.9)
    assert chat(gw, "What is the capital of France?").headers["x-cache"] == "MISS"
    hit = chat(gw, "Tell me the capital of France, please.")
    assert hit.headers["x-cache"] == "HIT;SEMANTIC"
    assert fake.embed_calls() == []


def test_cached_hit_and_bad_json_body(gw):
    r = gw.post("/v1/chat/completions", headers=auth(), content=b"{nope")
    assert r.status_code == 400
