"""Routing heuristics and failover-chain resolution."""

from app.providers import ProviderRegistry
from app.router import Router, RoutingConfig, classify, detect_code
from app.schemas import ChatMessage

PROVIDERS = {
    "providers": {
        "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "api_key_env": "NVIDIA_API_KEY"},
        "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": "OLLAMA_API_KEY", "default_api_key": "ollama"},
    },
    "models": {
        "meta/llama-3.3-70b-instruct": {"provider": "nvidia"},
        "meta/llama-3.1-8b-instruct": {"provider": "nvidia"},
        "ollama/llama3.1": {"provider": "ollama", "upstream_model": "llama3.1"},
    },
}

ROUTING = {
    "auto": {
        "long_prompt_chars": 100,
        "models": {"cheap": "meta/llama-3.1-8b-instruct", "strong": "meta/llama-3.3-70b-instruct"},
        "rules": [
            {"if": "code", "use": "strong"},
            {"if": "json_mode", "use": "strong"},
            {"if": "long_prompt", "use": "strong"},
            {"if": "default", "use": "cheap"},
        ],
    },
    "aliases": {"gpt-3.5-turbo": "meta/llama-3.1-8b-instruct"},
    "fallbacks": {"meta/llama-3.3-70b-instruct": ["meta/llama-3.1-8b-instruct", "ollama/llama3.1"]},
}


def make_router() -> Router:
    registry = ProviderRegistry.from_dict(PROVIDERS)
    routing = RoutingConfig.from_dict(ROUTING)
    return Router(routing, registry)


def msgs(content: str, role: str = "user") -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content)]


def test_detect_code():
    assert detect_code("```python\nprint(1)\n```")
    assert detect_code("def add(a, b):")
    assert detect_code("SELECT id FROM users")
    assert not detect_code("What is the tallest mountain in the world?")


def test_classify_signals():
    cfg = make_router().config.auto
    signals = classify(msgs("write a function def foo(x):"), False, cfg)
    assert signals["code"] is True
    plain = classify(msgs("hi"), False, cfg)
    assert plain["code"] is False
    assert plain["long_prompt"] is False


def test_auto_short_prompt_routes_cheap():
    router = make_router()
    model, decision = router.resolve_model("auto", msgs("what is 2+2?"), False)
    assert model == "meta/llama-3.1-8b-instruct"
    assert decision["tier"] == "cheap"
    assert decision["reason"] == "default"


def test_auto_code_routes_strong():
    router = make_router()
    model, decision = router.resolve_model("auto", msgs("```js\nconsole.log(1)\n```"), False)
    assert model == "meta/llama-3.3-70b-instruct"
    assert decision["reason"] == "code"


def test_auto_json_mode_routes_strong():
    router = make_router()
    model, decision = router.resolve_model("auto", msgs("give me data"), True)
    assert model == "meta/llama-3.3-70b-instruct"
    assert decision["reason"] == "json_mode"


def test_auto_long_prompt_routes_strong():
    router = make_router()
    long_text = "word " * 40  # > 100 chars
    model, decision = router.resolve_model("auto", msgs(long_text), False)
    assert model == "meta/llama-3.3-70b-instruct"
    assert decision["reason"] == "long_prompt"


def test_alias_resolution():
    router = make_router()
    model, decision = router.resolve_model("gpt-3.5-turbo", msgs("hi"), False)
    assert model == "meta/llama-3.1-8b-instruct"
    assert decision["reason"] == "alias"


def test_exact_model_passthrough():
    router = make_router()
    model, decision = router.resolve_model("meta/llama-3.3-70b-instruct", msgs("hi"), False)
    assert model == "meta/llama-3.3-70b-instruct"
    assert decision["reason"] == "exact"


def test_fallback_chain_crosses_providers():
    router = make_router()
    chain = router.resolve_chain("meta/llama-3.3-70b-instruct")
    assert [t.upstream_model for t in chain] == [
        "meta/llama-3.3-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "llama3.1",
    ]
    assert chain[0].provider.name == "nvidia"
    assert chain[2].provider.name == "ollama"


def test_fallback_chain_single_when_no_fallbacks():
    router = make_router()
    chain = router.resolve_chain("meta/llama-3.1-8b-instruct")
    assert [t.gateway_model for t in chain] == ["meta/llama-3.1-8b-instruct"]
