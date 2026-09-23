"""Two-layer cache: exact + semantic (embeddings mocked, no network)."""

from app.cache import SemanticCache, context_key, estimated_savings, exact_key, semantic_text
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


def test_semantic_text_is_the_final_user_turn():
    body = {
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
    }
    # The system prompt and earlier turns are pinned by context_key instead.
    assert semantic_text(body) == "second"
    assert semantic_text({"messages": [{"role": "system", "content": "only system"}]}) == ""
    multimodal = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "this"},
    ]}]}
    assert semantic_text(multimodal) == "describe this"


def test_context_key_ignores_only_the_final_user_text():
    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"}]
    k1 = context_key("m", {"messages": base + [{"role": "user", "content": "follow up"}]})
    k2 = context_key("m", {"messages": base + [{"role": "user", "content": "a different follow up"}]})
    assert k1 == k2
    other_history = [{"role": "system", "content": "s"}, {"role": "user", "content": "q2"},
                     {"role": "assistant", "content": "a1"}, {"role": "user", "content": "follow up"}]
    assert context_key("m", {"messages": other_history}) != k1
    body = {"messages": base + [{"role": "user", "content": "follow up"}]}
    for field, value in [("response_format", {"type": "json_object"}), ("temperature", 0.9),
                         ("max_tokens", 5), ("tools", [{"type": "function"}]), ("seed", 7)]:
        assert context_key("m", {**body, field: value}) != k1, field
    assert context_key("other-model", body) != k1
    # Fields that do not shape the output do not split the context.
    assert context_key("m", {**body, "user": "u1", "stream": True}) == k1


async def test_semantic_hit_requires_identical_context():
    cache = make_cache()
    history = [{"role": "user", "content": "talk about Spain"}, {"role": "assistant", "content": "ok"}]
    await cache.store("m", {"messages": history + [{"role": "user", "content": "France?"}]}, RESP, allow_semantic=True)
    other = [{"role": "user", "content": "talk about cheese"}, {"role": "assistant", "content": "ok"}]
    assert await cache.lookup("m", {"messages": other + [{"role": "user", "content": "France!"}]},
                              allow_semantic=True) is None
    hit = await cache.lookup("m", {"messages": history + [{"role": "user", "content": "France!"}]},
                             allow_semantic=True)
    assert hit is not None and hit.kind == "SEMANTIC"


async def test_failing_embedder_degrades_to_exact_only():
    calls = {"n": 0}

    async def broken(text: str) -> list[float]:
        calls["n"] += 1
        raise RuntimeError("embedder down")

    cache = SemanticCache(Database(":memory:"), broken, ttl_seconds=60, similarity_threshold=0.9)
    probe = await cache.probe("m", req("hello"), allow_semantic=True)
    assert probe.hit is None and probe.embed_failed
    await cache.store("m", req("hello"), RESP, allow_semantic=True, embed=not probe.embed_failed)
    assert calls["n"] == 1
    await cache.store("m", req("other"), RESP, allow_semantic=True)  # store's own embed fails too
    assert calls["n"] == 2
    assert (await cache.lookup("m", req("other"), allow_semantic=False)).kind == "EXACT"


async def test_probe_vector_is_reused_by_store():
    calls = {"n": 0}

    async def counting(text: str) -> list[float]:
        calls["n"] += 1
        return await topic_embedder(text)

    cache = SemanticCache(Database(":memory:"), counting, ttl_seconds=60, similarity_threshold=0.9)
    probe = await cache.probe("m", req("France"), allow_semantic=True)
    await cache.store("m", req("France"), RESP, allow_semantic=True, vector=probe.vector)
    assert calls["n"] == 1
    assert cache.stats()["semantic_entries"] == 1


def test_estimated_savings_math():
    saving = estimated_savings(hits=100, avg_prompt_tokens=500, avg_completion_tokens=200,
                               input_per_1m=0.9, output_per_1m=0.9)
    expected = round(100 * (500 / 1_000_000 * 0.9 + 200 / 1_000_000 * 0.9), 6)
    assert saving == expected
