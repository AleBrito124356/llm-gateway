"""HTTP surface: auth, validation, routing errors, caching rules, tool calls."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.main import build_app
from app.providers import ConfigError

from .conftest import CHEAP_KEY, KEY, PROVIDERS, auth, chat, sse_events


def test_missing_and_invalid_gateway_keys(gw):
    no_key = gw.post("/v1/chat/completions", json={"model": "big", "messages": []})
    assert no_key.status_code == 401
    assert no_key.json()["error"]["type"] == "authentication_error"
    assert chat(gw, "hi", key="sk-wrong").status_code == 401
    raw_header = gw.post("/v1/chat/completions", headers={"Authorization": KEY},
                         json={"model": "big", "messages": [{"role": "user", "content": "hi"}]})
    assert raw_header.status_code == 200  # a bare key without "Bearer" is accepted too
    assert gw.post("/v1/embeddings", json={"model": "embed", "input": "x"}).status_code == 401


@pytest.mark.parametrize("body", [b"{not json", b"[1, 2, 3]", b'{"model": "big"}', b'{"messages": []}'])
def test_malformed_bodies_are_400(gw, body):
    r = gw.post("/v1/chat/completions", headers={**auth(), "content-type": "application/json"}, content=body)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_malformed_embedding_bodies_are_400(gw):
    assert gw.post("/v1/embeddings", headers=auth(), json={"model": "embed"}).status_code == 400
    assert gw.post("/v1/embeddings", headers=auth(), content=b"nope").status_code == 400


def test_unknown_models_are_404(gw):
    r = chat(gw, "hi", model="no-such-model")
    assert r.status_code == 404 and r.json()["error"]["type"] == "model_not_found"
    assert "x-ratelimit-limit" in r.headers
    e = gw.post("/v1/embeddings", headers=auth(), json={"model": "nope", "input": "x"})
    assert e.status_code == 404


def test_model_allow_lists_are_403(gw, fake):
    r = chat(gw, "hi", model="big", key=CHEAP_KEY)
    assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
    assert chat(gw, "hi", model="gpt-3.5-turbo", key=CHEAP_KEY).status_code == 200  # alias to small
    e = gw.post("/v1/embeddings", headers=auth(CHEAP_KEY), json={"model": "local-embed", "input": "x"})
    assert e.status_code == 403
    assert all(c["body"]["model"] != "meta/big" for c in fake.chat_calls())


def test_auto_routing_headers(gw):
    cheap = chat(gw, "what is 2+2?", model="auto")
    assert cheap.headers["x-gateway-model"] == "small" and cheap.headers["x-gateway-route"] == "default"
    strong = chat(gw, "fix this: def f(x): return y", model="auto")
    assert strong.headers["x-gateway-model"] == "big" and strong.headers["x-gateway-route"] == "code"
    exact = chat(gw, "hi", model="big")
    assert "x-gateway-route" not in exact.headers


def test_route_level_cache_toggles(gw, fake):
    chat(gw, "not cached", model="nocache")
    assert chat(gw, "not cached", model="nocache").headers["x-cache"] == "MISS"
    # semantic_cache: false still allows exact hits, but never fuzzy ones.
    chat(gw, "Local question here", model="local-chat")
    assert chat(gw, "Local question here", model="local-chat").headers["x-cache"] == "HIT;EXACT"
    assert chat(gw, "local question here!", model="local-chat").headers["x-cache"] == "MISS"


def test_global_cache_switches(make_gateway, fake):
    gw = make_gateway(cache_enabled=False)
    chat(gw, "same")
    assert chat(gw, "same").headers["x-cache"] == "MISS"
    gw2 = make_gateway(semantic_cache_enabled=False)
    chat(gw2, "What is the capital of France?")
    assert chat(gw2, "what is the capital of france").headers["x-cache"] == "MISS"
    assert fake.embed_calls() == []


TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]


def tool_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    call = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    if body.get("stream"):
        chunk = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                 "choices": [{"index": 0, "delta": {"tool_calls": [{**call, "index": 0}]}, "finish_reason": "tool_calls"}]}
        return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
    return httpx.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 1, "model": body["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                     "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    })


def test_tool_calls_skip_the_semantic_layer_and_replay_intact(make_gateway):
    gw = make_gateway(transport=httpx.MockTransport(tool_response))
    first = chat(gw, "look it up", tools=TOOLS)
    assert first.json()["choices"][0]["finish_reason"] == "tool_calls"
    # Exact repeats may hit; the replayed stream still carries the tool call.
    replay = gw.post("/v1/chat/completions", headers=auth(), json={
        "model": "big", "stream": True, "tools": TOOLS, "messages": [{"role": "user", "content": "look it up"}],
    })
    assert replay.headers["x-cache"] == "HIT;EXACT"
    events = [e for e in sse_events(replay) if isinstance(e, dict)]
    calls = [e["choices"][0]["delta"]["tool_calls"] for e in events if e["choices"][0]["delta"].get("tool_calls")]
    assert calls[0][0]["function"]["name"] == "lookup" and calls[0][0]["index"] == 0
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    # A paraphrase with tools is never served from the semantic layer.
    assert chat(gw, "Look it up!", tools=TOOLS).headers["x-cache"] == "MISS"


def test_streamed_tool_calls_are_not_cached(make_gateway):
    gw = make_gateway(transport=httpx.MockTransport(tool_response))
    body = {"model": "big", "stream": True, "tools": TOOLS, "messages": [{"role": "user", "content": "stream tool"}]}
    assert gw.post("/v1/chat/completions", headers=auth(), json=body).status_code == 200
    assert gw.post("/v1/chat/completions", headers=auth(), json=body).headers["x-cache"] == "MISS"


def test_require_auth_false_serves_anonymous_callers(make_gateway):
    gw = make_gateway(require_auth=False)
    r = gw.post("/v1/chat/completions", json={"model": "big", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    usage = gw.get("/admin/usage?format=json").json()
    assert usage["by_key"][0]["virtual_key"] == "anonymous"


def test_usage_without_upstream_usage_block_is_estimated(make_gateway):
    def no_usage(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0], "index": 0}]})
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant",
                                                                              "content": "x" * 40}}],
                                          "model": body["model"]})

    gw = make_gateway(transport=httpx.MockTransport(no_usage))
    assert chat(gw, "y" * 80).status_code == 200
    assert gw.post("/v1/embeddings", headers=auth(), json={"model": "embed", "input": ["a" * 40, "b"]}).status_code == 200
    rows = {r["model"]: r for r in gw.get("/admin/usage?format=json").json()["by_model"]}
    assert rows["big"]["total_tokens"] == 20 + 10  # ~4 chars per token
    assert rows["embed"]["total_tokens"] == 10


def test_missing_config_files_raise_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="CONFIG_DIR"):
        build_app(Settings.from_env(env={}, config_dir=str(tmp_path)))
    (tmp_path / "providers.yaml").write_text(json.dumps(PROVIDERS), encoding="utf-8")
    with pytest.raises(ConfigError, match="routing file not found"):
        build_app(Settings.from_env(env={}, config_dir=str(tmp_path)))
