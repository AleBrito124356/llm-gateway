"""Two-layer response cache.

Layer 1 - exact: hash the request fields that affect the output. Identical
requests return the stored response with zero upstream cost.

Layer 2 - semantic: embed the *final user turn*, and if a previous request with
the same **context** is within ``similarity_threshold`` cosine similarity, reuse
its answer. This catches paraphrases ("what's the capital of France?" vs "tell
me the capital of France, please") that an exact hash misses.

The context is everything except the final user turn's text: the model, the
system prompt, every earlier turn, and every output-shaping parameter
(``response_format``, ``tools``, ``max_tokens``, ``temperature``, ``stop``,
``seed`` ...). It is hashed into ``context_key`` and a semantic match must share
it exactly, so two different conversations that both end in "Give me an
example" never trade answers, and a JSON-mode request is never served a prose
answer that was cached for a plain request.

Both layers honor a TTL and a per-route toggle; the table is purged of expired
rows and capped at ``max_entries`` periodically. A failing embedder degrades the
cache to exact-only instead of disabling it. Hits are flagged in the ``X-Cache``
response header. See ``estimated_savings`` for the cost math and the README for
the honest caveats about *when not to cache*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import numpy as np

from .db import Database

logger = logging.getLogger("gateway.cache")

# An async function that turns a string into an embedding vector.
Embedder = Callable[[str], Awaitable[list[float]]]

# Fields that change the model output. Everything else (stream, user, ...) is
# excluded so a streaming and non-streaming call for the same content share a hit.
_KEYED_FIELDS = (
    "messages",
    "temperature",
    "top_p",
    "n",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "functions",
    "function_call",
    "reasoning_effort",
    "seed",
)

# Stands in for the final user turn's text inside the context key.
_FINAL_TURN_PLACEHOLDER = "\x00final-user-turn\x00"

# At most this many same-context candidates are scored per semantic lookup.
DEFAULT_SCAN_LIMIT = 2000


@dataclass
class CacheHit:
    response: dict[str, Any]
    kind: str  # "EXACT" or "SEMANTIC"
    similarity: float = 1.0


@dataclass
class CacheProbe:
    """Result of a lookup, plus what ``store`` needs to avoid re-embedding."""

    hit: Optional[CacheHit]
    vector: Optional[np.ndarray] = None
    embed_failed: bool = False


def _keyed_payload(model: str, body: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model}
    for key in _KEYED_FIELDS:
        if key in body and body[key] is not None:
            payload[key] = body[key]
    return payload


def _dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_request(model: str, body: dict[str, Any]) -> str:
    return _dumps(_keyed_payload(model, body))


def exact_key(model: str, body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_request(model, body).encode("utf-8")).hexdigest()


def _last_user_index(messages: list[Any]) -> Optional[int]:
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return index
    return None


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(
            str(c.get("text", "")) for c in content if isinstance(c, dict) and c.get("type", "text") == "text"
        )
    return content if isinstance(content, str) else ""


def context_key(model: str, body: dict[str, Any]) -> str:
    """Hash of the request with the final user turn's text blanked out.

    Two requests may share a semantic hit only if everything else about them is
    identical: model, system prompt, conversation history and output params.
    """
    payload = _keyed_payload(model, body)
    messages = list(body.get("messages") or [])
    index = _last_user_index(messages)
    if index is not None:
        masked = dict(messages[index])
        masked["content"] = _FINAL_TURN_PLACEHOLDER
        messages[index] = masked
    payload["messages"] = messages
    return hashlib.sha256(_dumps(payload).encode("utf-8")).hexdigest()


def semantic_text(body: dict[str, Any]) -> str:
    """The string we embed: the final user turn.

    The system prompt and earlier turns are pinned exactly by ``context_key``,
    so embedding them too would only dilute the similarity of the part that
    actually varies (a long shared system prompt makes every question look alike).
    """
    messages = body.get("messages") or []
    index = _last_user_index(messages)
    if index is None:
        return ""
    return _content_text(messages[index].get("content")).strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticCache:
    def __init__(
        self,
        db: Database,
        embedder: Optional[Embedder],
        *,
        ttl_seconds: int,
        similarity_threshold: float,
        semantic_enabled: bool = True,
        max_entries: int = 0,
        maintenance_interval_seconds: float = 300.0,
        scan_limit: int = DEFAULT_SCAN_LIMIT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.semantic_enabled = semantic_enabled
        self.max_entries = max_entries
        self.maintenance_interval_seconds = maintenance_interval_seconds
        self.scan_limit = scan_limit
        self.clock = clock
        self._last_maintenance = clock()

    def _semantic_active(self, allow_semantic: bool) -> bool:
        return self.semantic_enabled and allow_semantic and self.embedder is not None

    async def _embed(self, text: str) -> np.ndarray:
        assert self.embedder is not None
        return np.asarray(await self.embedder(text), dtype=np.float32)

    async def probe(
        self,
        model: str,
        body: dict[str, Any],
        *,
        allow_semantic: bool,
    ) -> CacheProbe:
        """Look both layers up; also return the query vector for ``store``."""
        now = self.clock()

        # Layer 1: exact hash.
        row = self.db.query_one(
            "SELECT response_json FROM cache WHERE exact_key = ? AND expires_at > ? "
            "ORDER BY id DESC LIMIT 1",
            (exact_key(model, body), now),
        )
        if row is not None:
            return CacheProbe(CacheHit(json.loads(row["response_json"]), "EXACT", 1.0))

        # Layer 2: semantic, restricted to requests with the identical context.
        if not self._semantic_active(allow_semantic):
            return CacheProbe(None)
        query_text = semantic_text(body)
        if not query_text:
            return CacheProbe(None)
        try:
            query_vec = await self._embed(query_text)
        except Exception as exc:  # noqa: BLE001 - a broken embedder must not break requests
            logger.warning("semantic cache disabled for this request, embedder failed: %s", exc)
            return CacheProbe(None, embed_failed=True)

        rows = self.db.query(
            "SELECT response_json, embedding FROM cache "
            "WHERE route = ? AND context_key = ? AND expires_at > ? AND embedding IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (model, context_key(model, body), now, self.scan_limit),
        )
        best: Optional[CacheHit] = None
        best_sim = self.similarity_threshold
        for candidate in rows:
            vec = np.frombuffer(candidate["embedding"], dtype=np.float32)
            if vec.shape != query_vec.shape:
                continue
            sim = cosine(query_vec, vec)
            if sim >= best_sim:
                best_sim = sim
                best = CacheHit(json.loads(candidate["response_json"]), "SEMANTIC", sim)
        return CacheProbe(best, vector=query_vec)

    async def lookup(
        self,
        model: str,
        body: dict[str, Any],
        *,
        allow_semantic: bool,
    ) -> Optional[CacheHit]:
        return (await self.probe(model, body, allow_semantic=allow_semantic)).hit

    async def store(
        self,
        model: str,
        body: dict[str, Any],
        response: dict[str, Any],
        *,
        allow_semantic: bool,
        vector: Optional[np.ndarray] = None,
        embed: bool = True,
    ) -> None:
        """Store a response. The exact entry is written even if embedding fails.

        Pass the ``vector`` from ``probe`` to avoid a second embedding call, or
        ``embed=False`` when the probe already saw the embedder fail.
        """
        now = self.clock()
        embedding_blob: Optional[bytes] = None
        if self._semantic_active(allow_semantic):
            if vector is None and embed:
                text = semantic_text(body)
                if text:
                    try:
                        vector = await self._embed(text)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("storing exact-only cache entry, embedder failed: %s", exc)
                        vector = None
            if vector is not None:
                embedding_blob = np.asarray(vector, dtype=np.float32).tobytes()
        self.db.execute(
            "INSERT INTO cache (exact_key, context_key, route, model, embedding, request_json, "
            "response_json, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exact_key(model, body),
                context_key(model, body),
                model,
                model,
                embedding_blob,
                canonical_request(model, body),
                json.dumps(response, ensure_ascii=False),
                now,
                now + self.ttl_seconds,
            ),
        )
        self.maybe_maintain()

    # ------------------------------------------------------------ housekeeping

    def purge_expired(self) -> int:
        """Delete expired rows; returns how many were removed."""
        return self.db.execute_count("DELETE FROM cache WHERE expires_at <= ?", (self.clock(),))

    def enforce_max_entries(self) -> int:
        """Evict the oldest rows beyond ``max_entries``; returns how many."""
        if self.max_entries <= 0:
            return 0
        return self.db.execute_count(
            "DELETE FROM cache WHERE id NOT IN (SELECT id FROM cache ORDER BY id DESC LIMIT ?)",
            (self.max_entries,),
        )

    def maintain(self) -> dict[str, int]:
        self._last_maintenance = self.clock()
        expired = self.purge_expired()
        evicted = self.enforce_max_entries()
        if expired or evicted:
            logger.info("cache maintenance: purged %d expired, evicted %d over cap", expired, evicted)
        return {"expired": expired, "evicted": evicted}

    def maybe_maintain(self) -> Optional[dict[str, int]]:
        if self.clock() - self._last_maintenance >= self.maintenance_interval_seconds:
            return self.maintain()
        return None

    def stats(self) -> dict[str, Any]:
        row = self.db.query_one(
            "SELECT COUNT(*) AS entries, SUM(embedding IS NOT NULL) AS semantic_entries, "
            "SUM(expires_at <= ?) AS expired FROM cache",
            (self.clock(),),
        )
        return {
            "entries": int(row["entries"] or 0) if row else 0,
            "semantic_entries": int(row["semantic_entries"] or 0) if row else 0,
            "expired_pending_purge": int(row["expired"] or 0) if row else 0,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
        }


def estimated_savings(
    hits: int,
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    input_per_1m: float,
    output_per_1m: float,
) -> float:
    """USD not spent upstream because ``hits`` requests were served from cache.

    Each cache hit avoids one full generation, so the saving is the price of the
    tokens that would have been billed:

        saving = hits * (prompt/1e6 * input_price + completion/1e6 * output_price)
    """
    per_hit = (
        avg_prompt_tokens / 1_000_000 * input_per_1m
        + avg_completion_tokens / 1_000_000 * output_per_1m
    )
    return round(hits * per_hit, 6)
