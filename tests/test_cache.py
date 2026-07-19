"""Two-layer cache: exact + semantic (embeddings mocked, no network)."""

from app.cache import SemanticCache, estimated_savings, exact_key, semantic_text
from app.db import Database


async def topic_embedder(text: str) -> list[float]:
    """Deterministic stand-in: same topic -> identical vector."""
    lowered = text.lower()
    if "france" in lowered:
        return [1.0, 0.0, 0.0]
    if "spain" in lowered:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def make_cache(ttl: int = 3600, semantic: bool = True) -> SemanticCache:
    db = Database(":memory:")
    return SemanticCache(
        db, topic_embedder, ttl_seconds=ttl, similarity_threshold=0.9, semantic_enabled=semantic
    )


RESP = {
    "id": "chatcmpl-x",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
}


def req(content: str) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


async def test_exact_miss_then_hit():
    cache = make_cache()
    assert await cache.lookup("m", req("hello"), allow_semantic=False) is None
    await cache.store("m", req("hello"), RESP, allow_semantic=False)
    hit = await cache.lookup("m", req("hello"), allow_semantic=False)
    assert hit is not None
    assert hit.kind == "EXACT"
    assert hit.response["choices"][0]["message"]["content"] == "Paris"


async def test_semantic_hit_on_paraphrase():
    cache = make_cache()
    await cache.store("m", req("What is the capital of France?"), RESP, allow_semantic=True)
    # Different wording -> exact miss, but same topic -> semantic hit.
    hit = await cache.lookup("m", req("Tell me the capital of France please"), allow_semantic=True)
    assert hit is not None
    assert hit.kind == "SEMANTIC"
    assert hit.similarity >= 0.9


async def test_semantic_miss_on_different_topic():
    cache = make_cache()
    await cache.store("m", req("capital of France?"), RESP, allow_semantic=True)
    assert await cache.lookup("m", req("capital of Spain?"), allow_semantic=True) is None


async def test_semantic_disabled_flag():
    cache = make_cache(semantic=False)
    await cache.store("m", req("France one"), RESP, allow_semantic=True)
    assert await cache.lookup("m", req("France two"), allow_semantic=True) is None


async def test_allow_semantic_false_skips_layer_two():
    cache = make_cache()
    await cache.store("m", req("France A"), RESP, allow_semantic=True)
    # Layer 2 explicitly disallowed for this request -> only exact would hit.
    assert await cache.lookup("m", req("France B"), allow_semantic=False) is None


async def test_ttl_expiry():
    cache = make_cache(ttl=-1)  # already expired on write
    await cache.store("m", req("hello"), RESP, allow_semantic=False)
    assert await cache.lookup("m", req("hello"), allow_semantic=False) is None


async def test_route_isolation():
    cache = make_cache()
    await cache.store("model-a", req("France"), RESP, allow_semantic=True)
    # Same text, different route -> no semantic bleed across models.
    assert await cache.lookup("model-b", req("France"), allow_semantic=True) is None


def test_exact_key_is_stable_and_content_sensitive():
    assert exact_key("m", req("a")) == exact_key("m", req("a"))
    assert exact_key("m", req("a")) != exact_key("m", req("b"))
    assert exact_key("m", req("a")) != exact_key("other", req("a"))


def test_semantic_text_uses_system_and_last_user():
    body = {
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
    }
    text = semantic_text(body)
    assert "be terse" in text
    assert "second" in text
    assert "first" not in text


def test_estimated_savings_math():
    saving = estimated_savings(hits=100, avg_prompt_tokens=500, avg_completion_tokens=200,
                               input_per_1m=0.9, output_per_1m=0.9)
    expected = round(100 * (500 / 1_000_000 * 0.9 + 200 / 1_000_000 * 0.9), 6)
    assert saving == expected
