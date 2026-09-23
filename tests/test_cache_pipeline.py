"""Regression tests for the cache and OpenAI-contract bugs found in the audit.

Each test drives the real FastAPI app through HTTP with a recording fake
upstream, so it checks what a client and the upstream actually see.
"""

from __future__ import annotations

import sqlite3

from app.cache import SemanticCache
from app.db import Database

from .conftest import BROKE_KEY, KEY, LIMITED_KEY, auth, chat, sse_events

CONV_A = [
    {"role": "user", "content": "Tell me about Python lists"},
    {"role": "assistant", "content": "Lists are ordered, mutable sequences."},
    {"role": "user", "content": "Give me an example"},
]
CONV_B = [
    {"role": "user", "content": "Tell me about SQL joins"},
    {"role": "assistant", "content": "Joins combine rows from two tables."},
    {"role": "user", "content": "Give me an example"},
]


def test_paraphrase_hits_semantic_layer(gw):
    first = chat(gw, "What is the capital of France?")
    assert first.headers["x-cache"] == "MISS"
    again = chat(gw, "what is the capital of france")  # different bytes, same words
    assert again.status_code == 200
    assert again.headers["x-cache"] == "HIT;SEMANTIC"
    assert again.json() == first.json()


def test_semantic_cache_does_not_mix_conversations(gw, fake):
    a = chat(gw, CONV_A)
    b = chat(gw, CONV_B)
    assert a.headers["x-cache"] == "MISS"
    # Same final turn, different history: must NOT be served conversation A's answer.
    assert b.headers["x-cache"] == "MISS"
    assert len(fake.chat_calls()) == 2
    # The exact same conversation still hits.
    assert chat(gw, CONV_B).headers["x-cache"] == "HIT;EXACT"


def test_semantic_cache_respects_response_format(gw, fake):
    prose = chat(gw, "List 3 colors", model="small")
    json_mode = chat(gw, "List 3 colors", model="small", response_format={"type": "json_object"})
    assert prose.headers["x-cache"] == "MISS"
    assert json_mode.headers["x-cache"] == "MISS"
    assert len(fake.chat_calls()) == 2


def test_semantic_cache_respects_system_prompt_and_max_tokens(gw):
    chat(gw, [{"role": "system", "content": "Answer in French."}, {"role": "user", "content": "Say hi"}])
    other_system = chat(gw, [{"role": "system", "content": "Answer in German."}, {"role": "user", "content": "say hi"}])
    assert other_system.headers["x-cache"] == "MISS"
    chat(gw, "Summarise the news", max_tokens=10)
    assert chat(gw, "summarise the news", max_tokens=500).headers["x-cache"] == "MISS"


def test_exact_cache_survives_a_failing_embedder(gw, fake):
    fake.fail_always("embedder", 400)
    fake.fail_always("nomic", 400)
    first = chat(gw, "Same question twice")
    second = chat(gw, "Same question twice")
    assert first.status_code == second.status_code == 200
    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT;EXACT"
    # The failed lookup embed is not repeated at store time.
    assert len(fake.embed_calls()) == 1


def test_one_embedding_call_per_miss(gw, fake):
    chat(gw, "How tall is Everest?")
    assert len(fake.embed_calls()) == 1  # lookup embeds, store reuses the vector
    chat(gw, "How tall is Everest?")  # exact hit: no embedding at all
    assert len(fake.embed_calls()) == 1


def test_cache_embedder_sends_extra_body(gw, fake):
    chat(gw, "Embed me please")
    call = fake.embed_calls()[0]
    assert call["body"]["input_type"] == "query"
    assert call["body"]["model"] == "embedder"


def test_embeddings_endpoint_merges_extra_body_under_caller_fields(gw, fake):
    r = gw.post("/v1/embeddings", headers=auth(), json={
        "model": "text-embedding-3-small", "input": ["a doc"], "input_type": "passage",
    })
    assert r.status_code == 200
    assert fake.embed_calls()[-1]["body"]["input_type"] == "passage"  # caller wins
    assert r.headers["x-gateway-model"] == "embed"


def test_gateway_only_cache_field_is_not_forwarded(gw, fake):
    r = chat(gw, "Do not cache this", model="gpt-4", cache=False)
    assert r.status_code == 200
    assert r.headers["x-cache"] == "MISS"
    assert all("cache" not in call["body"] for call in fake.chat_calls())
    # cache:false really skips the cache on the way back, too.
    assert chat(gw, "Do not cache this", model="gpt-4", cache=False).headers["x-cache"] == "MISS"


def test_stream_hides_usage_chunk_unless_requested(gw, fake):
    r = gw.post("/v1/chat/completions", headers=auth(), json={
        "model": "small", "stream": True, "messages": [{"role": "user", "content": "stream please"}],
    })
    events = sse_events(r)
    assert r.status_code == 200
    assert events[-1] == "[DONE]"
    assert all(e["choices"] for e in events if isinstance(e, dict)), "usage-only chunk leaked"
    # The gateway still asked upstream for usage, for its own accounting.
    assert fake.chat_calls()[-1]["body"]["stream_options"]["include_usage"] is True

    r2 = gw.post("/v1/chat/completions", headers=auth(), json={
        "model": "small", "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "stream with usage"}],
    })
    usage_chunks = [e for e in sse_events(r2) if isinstance(e, dict) and e["choices"] == []]
    assert len(usage_chunks) == 1 and usage_chunks[0]["usage"]["total_tokens"] == 12


def test_cached_stream_replay_honours_include_usage(gw):
    chat(gw, "replay me")
    plain = gw.post("/v1/chat/completions", headers=auth(), json={
        "model": "big", "stream": True, "messages": [{"role": "user", "content": "replay me"}],
    })
    assert plain.headers["x-cache"] == "HIT;EXACT"
    events = [e for e in sse_events(plain) if isinstance(e, dict)]
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "meta/big says: replay me"
    assert all(e["choices"] for e in events)
    with_usage = gw.post("/v1/chat/completions", headers=auth(), json={
        "model": "big", "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "replay me"}],
    })
    tail = [e for e in sse_events(with_usage) if isinstance(e, dict)][-1]
    assert tail["choices"] == [] and tail["usage"]["completion_tokens"] == 5


def test_streamed_answer_is_cached_for_later_requests(gw, fake):
    gw.post("/v1/chat/completions", headers=auth(), json={
        "model": "small", "stream": True, "messages": [{"role": "user", "content": "cache my stream"}],
    })
    r = chat(gw, "cache my stream", model="small")
    assert r.headers["x-cache"] == "HIT;EXACT"
    assert r.json()["choices"][0]["message"]["content"] == "meta/small says: cache my stream"
    assert len(fake.chat_calls()) == 1


def test_429_carries_rate_limit_headers_and_computed_retry_after(gw):
    for _ in range(2):
        assert chat(gw, "hi", key=LIMITED_KEY).status_code == 200
    r = chat(gw, "hi", key=LIMITED_KEY)
    assert r.status_code == 429
    assert r.headers["x-ratelimit-limit"] == "2"
    assert r.headers["x-ratelimit-remaining"] == "0"
    assert int(r.headers["x-ratelimit-reset"]) > 0
    # rpm=2 refills one request every 30 s.
    assert 25 <= int(r.headers["retry-after"]) <= 30
    assert r.json()["error"]["type"] == "rate_limit_error"


def test_402_carries_budget_headers(gw):
    assert chat(gw, "spend some money", key=BROKE_KEY).status_code == 200
    r = chat(gw, "spend more", key=BROKE_KEY)
    assert r.status_code == 402
    assert r.headers["x-budget-limit"] == "0.0001"
    assert float(r.headers["x-budget-spent"]) > 0.0001
    assert r.headers["x-budget-remaining"] == "0.0000"
    assert int(r.headers["x-budget-reset"]) > 0
    assert int(r.headers["retry-after"]) >= 1
    assert "x-ratelimit-limit" in r.headers


def test_success_responses_carry_limit_headers(gw):
    r = chat(gw, "headers please", key=BROKE_KEY)
    assert r.headers["x-ratelimit-limit"] == "1000"
    assert r.headers["x-budget-remaining"] == "0.0001"
    unlimited = chat(gw, "no budget key", key=KEY)
    assert "x-budget-limit" not in unlimited.headers  # no budget -> no budget headers


def test_upstream_auth_failure_is_not_reported_as_caller_401(make_gateway, fake):
    gw = make_gateway(routing={"aliases": {}, "fallbacks": {}})
    fake.fail_always("meta/big", 401)
    r = chat(gw, "who am I", model="big")
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_error"


# ------------------------------------------------------------ storage level


def _old_schema_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT, exact_key TEXT NOT NULL, route TEXT NOT NULL,
            model TEXT NOT NULL, embedding BLOB, request_json TEXT NOT NULL,
            response_json TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


async def test_old_database_is_migrated_in_place(tmp_path):
    path = str(tmp_path / "old.db")
    _old_schema_db(path)
    db = Database(path)
    columns = {row["name"] for row in db.query("PRAGMA table_info(cache)")}
    assert "context_key" in columns
    cache = SemanticCache(db, None, ttl_seconds=60, similarity_threshold=0.9)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    await cache.store("m", body, {"choices": []}, allow_semantic=False)
    assert (await cache.lookup("m", body, allow_semantic=False)).kind == "EXACT"
    Database(path).close()  # re-opening an already-migrated file is a no-op


async def test_purge_and_cap_keep_the_table_bounded():
    now = [1000.0]
    db = Database(":memory:")
    cache = SemanticCache(
        db, None, ttl_seconds=10, similarity_threshold=0.9, max_entries=3,
        maintenance_interval_seconds=60, clock=lambda: now[0],
    )
    for i in range(5):
        await cache.store("m", {"messages": [{"role": "user", "content": f"q{i}"}]}, {"i": i}, allow_semantic=False)
    assert cache.stats()["entries"] == 5  # interval not reached yet
    now[0] += 61
    assert cache.maybe_maintain() == {"expired": 5, "evicted": 0}
    for i in range(5):
        await cache.store("m", {"messages": [{"role": "user", "content": f"n{i}"}]}, {"i": i}, allow_semantic=False)
    assert cache.maintain() == {"expired": 0, "evicted": 2}
    kept = await cache.lookup("m", {"messages": [{"role": "user", "content": "n4"}]}, allow_semantic=False)
    assert kept is not None and kept.response == {"i": 4}
    assert await cache.lookup("m", {"messages": [{"role": "user", "content": "n0"}]}, allow_semantic=False) is None
