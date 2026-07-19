"""Two-layer response cache.

Layer 1 - exact: hash the request fields that affect the output. Identical
requests return the stored response with zero upstream cost.

Layer 2 - semantic: embed ``system + last user message``, and if a previous
request to the same model is within ``similarity_threshold`` cosine distance,
reuse its answer. This catches paraphrases ("what's the capital of France?" vs
"tell me France's capital") that an exact hash misses.

Both layers honor a TTL and a per-route toggle. Hits are flagged in the
``X-Cache`` response header. See ``estimated_savings`` for the cost math and the
README for the honest caveats about *when not to cache*.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import numpy as np

from .db import Database

# An async function that turns a string into an embedding vector.
Embedder = Callable[[str], Awaitable[list[float]]]

# Fields that change the model output. Everything else (stream, user, ...) is
# excluded so a streaming and non-streaming call for the same content share a hit.
_KEYED_FIELDS = (
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "response_format",
    "tools",
    "tool_choice",
    "seed",
)


@dataclass
class CacheHit:
    response: dict[str, Any]
    kind: str  # "EXACT" or "SEMANTIC"
    similarity: float = 1.0


def canonical_request(model: str, body: dict[str, Any]) -> str:
    payload = {"model": model}
    for key in _KEYED_FIELDS:
        if key in body and body[key] is not None:
            payload[key] = body[key]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def exact_key(model: str, body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_request(model, body).encode("utf-8")).hexdigest()


def semantic_text(body: dict[str, Any]) -> str:
    """The string we embed: the system prompt plus the last user turn."""
    messages = body.get("messages") or []
    system = ""
    last_user = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(c.get("text", "")) for c in content if isinstance(c, dict)
            )
        content = content or ""
        if role == "system":
            system = content
        elif role == "user":
            last_user = content
    return (system + "\n" + last_user).strip()


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
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.semantic_enabled = semantic_enabled

    async def lookup(
        self,
        model: str,
        body: dict[str, Any],
        *,
        allow_semantic: bool,
    ) -> Optional[CacheHit]:
        now = time.time()

        # Layer 1: exact hash.
        row = self.db.query_one(
            "SELECT response_json FROM cache WHERE exact_key = ? AND expires_at > ? "
            "ORDER BY created_at DESC LIMIT 1",
            (exact_key(model, body), now),
        )
        if row is not None:
            return CacheHit(json.loads(row["response_json"]), "EXACT", 1.0)

        # Layer 2: semantic.
        if not (self.semantic_enabled and allow_semantic and self.embedder is not None):
            return None
        query_text = semantic_text(body)
        if not query_text:
            return None
        query_vec = np.asarray(await self.embedder(query_text), dtype=np.float32)
        rows = self.db.query(
            "SELECT response_json, embedding FROM cache "
            "WHERE route = ? AND expires_at > ? AND embedding IS NOT NULL",
            (model, now),
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
        return best

    async def store(
        self,
        model: str,
        body: dict[str, Any],
        response: dict[str, Any],
        *,
        allow_semantic: bool,
    ) -> None:
        now = time.time()
        embedding_blob: Optional[bytes] = None
        if self.semantic_enabled and allow_semantic and self.embedder is not None:
            text = semantic_text(body)
            if text:
                vec = np.asarray(await self.embedder(text), dtype=np.float32)
                embedding_blob = vec.tobytes()
        self.db.execute(
            "INSERT INTO cache (exact_key, route, model, embedding, request_json, "
            "response_json, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exact_key(model, body),
                model,
                model,
                embedding_blob,
                canonical_request(model, body),
                json.dumps(response, ensure_ascii=False),
                now,
                now + self.ttl_seconds,
            ),
        )

    def purge_expired(self) -> int:
        return self.db.execute("DELETE FROM cache WHERE expires_at <= ?", (time.time(),))


def estimated_savings(
    hits: int,
    avg_prompt_tokens: int,
    avg_completion_tokens: int,
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
