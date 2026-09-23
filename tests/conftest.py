"""Shared fixtures: a throwaway config dir, a recording fake upstream, and a
factory that boots the real FastAPI app against them. Nothing touches the network:
the app's upstream client is bound to an ``httpx.MockTransport``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import pytest
import yaml
from starlette.testclient import TestClient

from app.config import Settings
from app.main import build_app

KEY = "sk-test-main-key"
LIMITED_KEY = "sk-test-limited-key"
BROKE_KEY = "sk-test-broke-key"
CHEAP_KEY = "sk-test-cheap-only"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def auth(key: str = KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


PROVIDERS: dict[str, Any] = {
    "providers": {
        "nim": {"base_url": "https://nim.test/v1", "api_key_env": "TEST_NIM_KEY"},
        "local": {"base_url": "http://local.test/v1", "api_key_env": "TEST_LOCAL_KEY", "default_api_key": "local"},
        "keyless": {"base_url": "https://keyless.test/v1", "api_key_env": "TEST_UNSET_KEY"},
    },
    "models": {
        "big": {"provider": "nim", "upstream_model": "meta/big"},
        "small": {"provider": "nim", "upstream_model": "meta/small"},
        "local-chat": {"provider": "local", "upstream_model": "llama", "semantic_cache": False},
        "nocache": {"provider": "nim", "upstream_model": "meta/nocache", "cache": False},
        "keyless-chat": {"provider": "keyless", "upstream_model": "whatever"},
        "embed": {"provider": "nim", "upstream_model": "embedder", "extra_body": {"input_type": "query"}},
        "local-embed": {"provider": "local", "upstream_model": "nomic"},
    },
}

ROUTING: dict[str, Any] = {
    "auto": {
        "long_prompt_chars": 200,
        "models": {"cheap": "small", "strong": "big"},
        "rules": [
            {"if": "code", "use": "strong"},
            {"if": "json_mode", "use": "strong"},
            {"if": "long_prompt", "use": "strong"},
            {"if": "default", "use": "cheap"},
        ],
    },
    "aliases": {"gpt-4": "big", "gpt-3.5-turbo": "small", "text-embedding-3-small": "embed"},
    "fallbacks": {"big": ["small", "local-chat"], "keyless-chat": ["local-chat"], "embed": ["local-embed"]},
}

KEYS: dict[str, Any] = {
    "keys": [
        {"name": "main", "key_hash": sha(KEY), "rpm": 1000, "allowed_models": ["*"]},
        {"name": "limited", "key_hash": sha(LIMITED_KEY), "rpm": 2},
        {"name": "broke", "key_hash": sha(BROKE_KEY), "rpm": 1000, "monthly_budget_usd": 0.0001},
        {"name": "cheap-only", "key_hash": sha(CHEAP_KEY), "rpm": 1000, "monthly_budget_usd": 1.0,
         "allowed_models": ["small", "embed"]},
    ]
}

PRICING: dict[str, Any] = {
    "default": {"input_per_1m": 1.0, "output_per_1m": 1.0},
    "big": {"input_per_1m": 100.0, "output_per_1m": 100.0},
    "small": {"input_per_1m": 10.0, "output_per_1m": 10.0},
    "local-chat": {"input_per_1m": 0.0, "output_per_1m": 0.0},
}


def write_config(
    directory: Path,
    *,
    providers: Optional[dict] = None,
    routing: Optional[dict] = None,
    keys: Optional[dict] = None,
    pricing: Optional[dict] = None,
) -> None:
    (directory / "providers.yaml").write_text(yaml.safe_dump(providers or PROVIDERS), encoding="utf-8")
    (directory / "routing.yaml").write_text(yaml.safe_dump(routing or ROUTING), encoding="utf-8")
    (directory / "keys.yaml").write_text(yaml.safe_dump(keys or KEYS), encoding="utf-8")
    (directory / "pricing.json").write_text(json.dumps(pricing or PRICING), encoding="utf-8")


def bag_of_words(text: str, dims: int = 64) -> list[float]:
    """Case- and punctuation-insensitive embedding: same words -> same vector."""
    vec = [0.0] * dims
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeUpstream:
    """A scriptable OpenAI-compatible upstream that records every request.

    ``script(upstream_model, 503, 503)`` makes the next two calls for that model
    fail with 503 before normal answers resume. A scripted entry may also be a
    ``(status, headers)`` tuple, or an ``httpx.Response`` used verbatim.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.scripts: dict[str, list[Any]] = {}
        self.always: dict[str, int] = {}

    def script(self, model: str, *responses: Any) -> None:
        self.scripts.setdefault(model, []).extend(responses)

    def fail_always(self, model: str, status: int) -> None:
        self.always[model] = status

    def chat_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["path"].endswith("/chat/completions")]

    def embed_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["path"].endswith("/embeddings")]

    def _scripted(self, model: str) -> Optional[httpx.Response]:
        if model in self.always:
            return httpx.Response(self.always[model], json={"error": f"{model} is failing"})
        queue = self.scripts.get(model)
        if not queue:
            return None
        item = queue.pop(0)
        if isinstance(item, httpx.Response):
            return item
        status, headers = item if isinstance(item, tuple) else (item, {})
        return httpx.Response(status, json={"error": f"scripted {status}"}, headers=headers)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.calls.append({
            "host": request.url.host, "path": request.url.path, "body": body,
            "auth": request.headers.get("authorization"),
        })
        model = body.get("model", "")
        scripted = self._scripted(model)
        if scripted is not None:
            return scripted
        if request.url.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return httpx.Response(200, json={
                "object": "list", "model": model,
                "data": [{"object": "embedding", "index": i, "embedding": bag_of_words(str(t))}
                         for i, t in enumerate(inputs)],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            })
        last_user = next((m for m in reversed(body.get("messages", [])) if m.get("role") == "user"), {})
        text = last_user.get("content") if isinstance(last_user.get("content"), str) else ""
        answer = f"{model} says: {text}"
        if body.get("stream"):
            chunks = [
                {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model,
                 "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}]},
                {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            if (body.get("stream_options") or {}).get("include_usage"):
                chunks.append({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model,
                               "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}})
            sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "id": "chatcmpl-fake", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
        })


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    write_config(tmp_path)
    return tmp_path


@pytest.fixture
def make_gateway(tmp_path: Path, fake: FakeUpstream, monkeypatch: pytest.MonkeyPatch):
    """Factory: ``make_gateway(settings_overrides..., providers=..., routing=...)``.

    Returns an entered ``TestClient`` (lifespan running); closed at teardown.
    """
    monkeypatch.setenv("TEST_NIM_KEY", "nim-secret")
    monkeypatch.delenv("TEST_UNSET_KEY", raising=False)
    monkeypatch.delenv("TEST_LOCAL_KEY", raising=False)
    clients: list[TestClient] = []

    def make(
        *,
        providers: Optional[dict] = None,
        routing: Optional[dict] = None,
        keys: Optional[dict] = None,
        pricing: Optional[dict] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        **overrides: Any,
    ) -> TestClient:
        write_config(tmp_path, providers=providers, routing=routing, keys=keys, pricing=pricing)
        defaults: dict[str, Any] = dict(
            db_path=str(tmp_path / "gateway.db"),
            embed_model="embed",
            cache_similarity_threshold=0.95,
            backoff_base_seconds=0.0,
            backoff_cap_seconds=0.0,
            max_retries_per_target=1,
        )
        defaults.update(overrides)
        settings = Settings.from_env(env={}, config_dir=str(tmp_path), **defaults)
        app = build_app(settings, transport=transport or httpx.MockTransport(fake))
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def gw(make_gateway: Callable[..., TestClient]) -> TestClient:
    return make_gateway()


def chat(client: TestClient, content: Any = "hello", *, model: str = "big", key: str = KEY, **extra: Any):
    messages = content if isinstance(content, list) else [{"role": "user", "content": content}]
    return client.post("/v1/chat/completions", headers=auth(key), json={"model": model, "messages": messages, **extra})


def sse_events(response) -> list[Any]:
    """Parse a streamed response body into JSON events (``[DONE]`` kept as a str)."""
    events: list[Any] = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events
