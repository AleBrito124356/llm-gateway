"""Offline mock upstream: an in-process, OpenAI-compatible provider.

Declare a provider with ``type: mock`` in providers.yaml and every model routed
to it is answered locally, deterministically and for free. Nothing listens on
a socket: requests to a mock provider's host are served by ``MockUpstream``
through ``GatewayTransport``, which sits under the gateway's shared
``httpx.AsyncClient`` and hands every other host to the real network (or to an
injected transport in tests). That makes every gateway feature - routing, both
cache layers, retries, failover, the circuit breaker, limits, budgets and
accounting - demonstrable and testable with no API key and no network.

What a mock provider returns:

* ``/chat/completions``: a deterministic reply that quotes the last user turn
  (valid JSON in JSON mode, a tool call when ``tools`` are offered), honouring
  ``n`` and ``max_tokens``/``max_completion_tokens`` (``finish_reason: length``),
  with usage counts from a simple word-piece tokenizer. With ``stream: true``
  it sends proper SSE chunks, and a final usage-only chunk only when
  ``stream_options.include_usage`` is set - exactly like OpenAI.
* ``/embeddings``: ``hash_embedding`` vectors (``float`` or ``base64``).

Failure injection, per provider (see examples/demo/providers.yaml)::

    fail_status: 503      # answer every request with this status ...
    fail_first_n: 2       # ... or only the first N requests, then recover
    retry_after: 1        # add a Retry-After header to injected failures
    latency_ms: 150       # delay every response
    stream_delay_ms: 20   # delay between streamed chunks
    embedding_dims: 384   # vector size for /embeddings

``hash_embedding`` is also the ``local/hash`` embedder: set
``EMBED_MODEL=local/hash`` to get a working semantic cache with no embedding
provider at all (useful for Ollama-only setups).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import re
import time
from collections import deque
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx
import numpy as np

from .providers import Provider, ProviderRegistry

DEFAULT_DIMS = 384
LOCAL_HASH_MODEL = "local/hash"

_TOKEN_RE = re.compile(r"\s*\S+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Function words and politeness filler: dropping them lets "What is the capital
# of France?" and "Tell me the capital of France, please" embed identically.
STOPWORDS = frozenset(
    """
    a an the of to in on at by for from with about into over and or but nor so
    is are was were be been being am do does did done can could would should
    will shall may might must have has had i me my mine we us our you your it
    its this that these those there here what whats which who whom whose how
    please pls tell kindly just give show let lets some any
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Whitespace-attached word pieces; joining them restores the text."""
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    """Rough BPE-like count: a word costs one token per 4 characters."""
    return sum(max(1, math.ceil(len(piece.strip()) / 4)) for piece in tokenize(text))


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _features(text: str) -> list[tuple[str, float]]:
    words = [w for w in _WORD_RE.findall(text.casefold()) if len(w) > 1 or w.isdigit()]
    content = [_stem(w) for w in words if w not in STOPWORDS] or [_stem(w) for w in words]
    feats: list[tuple[str, float]] = [("w:" + w, 1.0) for w in content]
    feats += [(f"b:{a} {b}", 0.5) for a, b in zip(content, content[1:])]
    for word in content:
        padded = f"<{word}>"
        grams = [padded[i:i + 3] for i in range(len(padded) - 2)]
        feats += [("c:" + g, 0.6 / math.sqrt(len(grams))) for g in grams]
    return feats


def hash_embedding(text: str, dims: int = DEFAULT_DIMS) -> list[float]:
    """Deterministic, L2-normalised feature-hashing embedding.

    Features are content-word unigrams, bigrams and character trigrams, hashed
    with a signed hash into ``dims`` buckets. Paraphrases that share their
    content words score close to 1.0; different topics score low. It is no
    substitute for a neural embedder on real traffic, but it needs no model,
    no network and no GPU.
    """
    vec = np.zeros(dims, dtype=np.float64)
    for feature, weight in _features(text):
        h = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")
        vec[h % dims] += weight if (h >> 63) & 1 else -weight
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist() if norm else vec.tolist()


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def _snippet(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def _json_mode(body: dict[str, Any]) -> bool:
    fmt = body.get("response_format") or {}
    return isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema")


class MockUpstream:
    """Serves one ``type: mock`` provider. Keeps a bounded log of calls."""

    def __init__(self, provider: Provider, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        opts = provider.options
        self.name = provider.name
        self.fail_status: Optional[int] = opts.get("fail_status")
        self.fail_first_n: Optional[int] = opts.get("fail_first_n")
        if self.fail_first_n is not None and self.fail_status is None:
            self.fail_status = 503
        self.retry_after = opts.get("retry_after")
        self.latency_ms = float(opts.get("latency_ms", 0) or 0)
        self.stream_delay_ms = float(opts.get("stream_delay_ms", 0) or 0)
        self.embedding_dims = int(opts.get("embedding_dims", DEFAULT_DIMS))
        self.requests_seen = 0
        self.calls: deque[dict[str, Any]] = deque(maxlen=1000)
        self._sleep = sleep

    # ------------------------------------------------------------ dispatch

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        try:
            body = json.loads(request.content) if request.content else {}
        except ValueError:
            return self._error(400, "request body is not valid JSON", "invalid_request_error")
        self.calls.append({"method": request.method, "path": path, "body": body})
        if self.latency_ms:
            await self._sleep(self.latency_ms / 1000)
        if request.method == "GET" and path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.method != "POST":
            return self._error(405, f"{request.method} not supported", "invalid_request_error")
        self.requests_seen += 1
        if self._should_fail():
            headers = {"retry-after": f"{float(self.retry_after):g}"} if self.retry_after is not None else {}
            return self._error(
                int(self.fail_status or 503),
                f"mock provider '{self.name}' injected failure {self.requests_seen}",
                "mock_injected_failure",
                headers,
            )
        if not isinstance(body, dict):
            return self._error(400, "request body must be a JSON object", "invalid_request_error")
        if path.endswith("/chat/completions"):
            return self._chat(body)
        if path.endswith("/embeddings"):
            return self._embeddings(body)
        return self._error(404, f"unknown path {path}", "not_found_error")

    def _should_fail(self) -> bool:
        if self.fail_status is None:
            return False
        return self.fail_first_n is None or self.requests_seen <= self.fail_first_n

    @staticmethod
    def _error(status: int, message: str, err_type: str, headers: Optional[dict[str, str]] = None) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"message": message, "type": err_type, "code": status}}, headers=headers or {},
        )

    # ---------------------------------------------------------------- chat

    def _reply(self, model: str, body: dict[str, Any]) -> str:
        last_user = next(
            (_text(m.get("content")) for m in reversed(body["messages"])
             if isinstance(m, dict) and m.get("role") == "user"),
            "",
        )
        if _json_mode(body):
            return json.dumps({"model": model, "mock": True, "echo": _snippet(last_user, 80)})
        if not last_user.strip():
            return f"Mock answer from {model}."
        return (
            f'Mock answer from {model}: you asked "{_snippet(last_user)}". '
            "This reply is generated offline; no real model was called."
        )

    @staticmethod
    def _tool_call(body: dict[str, Any]) -> Optional[dict[str, Any]]:
        tools = body.get("tools") or []
        choice = body.get("tool_choice")
        if not tools or choice == "none":
            return None
        name = None
        if isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name")
        if name is None:
            first = tools[0] if isinstance(tools[0], dict) else {}
            name = (first.get("function") or {}).get("name", "tool")
        digest = hashlib.sha256(json.dumps(body["messages"], sort_keys=True).encode()).hexdigest()[:12]
        return {"id": f"call_{digest}", "type": "function", "function": {"name": name, "arguments": "{}"}}

    def _chat(self, body: dict[str, Any]) -> httpx.Response:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._error(400, "'messages' must be a non-empty list", "invalid_request_error")
        model = str(body.get("model") or "mock")
        n = max(1, int(body.get("n") or 1))
        limit = body.get("max_completion_tokens") or body.get("max_tokens")
        prompt_tokens = sum(count_tokens(_text(m.get("content")) if isinstance(m, dict) else "") + 4
                            for m in messages) + 3

        tool_call = self._tool_call(body)
        if tool_call is not None:
            content: Optional[str] = None
            finish = "tool_calls"
            completion_tokens = count_tokens(tool_call["function"]["name"]) + 3
        else:
            content, finish = self._truncate(self._reply(model, body), limit)
            completion_tokens = count_tokens(content)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens * n,
            "total_tokens": prompt_tokens + completion_tokens * n,
        }
        completion_id = "chatcmpl-mock-" + hashlib.sha256(
            json.dumps([model, messages], sort_keys=True, default=str).encode()
        ).hexdigest()[:20]
        created = int(time.time())

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            stream = _MockSSE(self._sse_chunks(completion_id, created, model, n, content, tool_call, finish,
                                                usage if include_usage else None), self.stream_delay_ms, self._sleep)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_call is not None:
            message["tool_calls"] = [tool_call]
        return httpx.Response(200, json={
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": i, "message": message, "finish_reason": finish} for i in range(n)],
            "usage": usage,
            "system_fingerprint": "mock",
        })

    @staticmethod
    def _truncate(text: str, limit: Any) -> tuple[str, str]:
        if not isinstance(limit, int) or limit <= 0:
            return text, "stop"
        kept, used = [], 0
        for piece in tokenize(text):
            cost = max(1, math.ceil(len(piece.strip()) / 4))
            if used + cost > limit:
                return "".join(kept), "length"
            kept.append(piece)
            used += cost
        return text, "stop"

    @staticmethod
    def _sse_chunks(
        completion_id: str,
        created: int,
        model: str,
        n: int,
        content: Optional[str],
        tool_call: Optional[dict[str, Any]],
        finish: str,
        usage: Optional[dict[str, int]],
    ) -> list[dict[str, Any]]:
        def chunk(choices: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
            return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": model, "system_fingerprint": "mock", "choices": choices, **extra}

        out: list[dict[str, Any]] = []
        for i in range(n):
            out.append(chunk([{"index": i, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
            if tool_call is not None:
                out.append(chunk([{"index": i, "delta": {"tool_calls": [{**tool_call, "index": 0}]},
                                   "finish_reason": None}]))
            else:
                pieces = tokenize(content or "")
                for start in range(0, len(pieces), 3):
                    text = "".join(pieces[start:start + 3])
                    out.append(chunk([{"index": i, "delta": {"content": text}, "finish_reason": None}]))
            out.append(chunk([{"index": i, "delta": {}, "finish_reason": finish}]))
        if usage is not None:
            out.append(chunk([], usage=usage))
        return out

    # ---------------------------------------------------------- embeddings

    def _embeddings(self, body: dict[str, Any]) -> httpx.Response:
        raw = body.get("input")
        if raw is None or raw == "" or raw == []:
            return self._error(400, "'input' is required", "invalid_request_error")
        if isinstance(raw, str):
            texts = [raw]
        elif isinstance(raw, list) and all(isinstance(x, int) for x in raw):
            texts = [" ".join(map(str, raw))]
        elif isinstance(raw, list):
            texts = [x if isinstance(x, str) else " ".join(map(str, x)) for x in raw]
        else:
            return self._error(400, "'input' must be a string or a list", "invalid_request_error")
        dims = int(body.get("dimensions") or self.embedding_dims)
        as_base64 = body.get("encoding_format") == "base64"
        data = []
        for index, text in enumerate(texts):
            vector = hash_embedding(text, dims)
            encoded: Any = (
                base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii") if as_base64 else vector
            )
            data.append({"object": "embedding", "index": index, "embedding": encoded})
        tokens = sum(count_tokens(t) for t in texts)
        return httpx.Response(200, json={
            "object": "list", "model": body.get("model", "mock"), "data": data,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        })


class _MockSSE(httpx.AsyncByteStream):
    def __init__(self, chunks: list[dict[str, Any]], delay_ms: float,
                 sleep: Callable[[float], Awaitable[None]]) -> None:
        self._chunks = chunks
        self._delay = delay_ms / 1000
        self._sleep = sleep

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            if self._delay:
                await self._sleep(self._delay)
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


# ------------------------------------------------------------- transports


def host_key(url: str | httpx.URL) -> str:
    return httpx.URL(url).netloc.decode("ascii").lower()


def build_mocks(registry: ProviderRegistry) -> dict[str, MockUpstream]:
    """One ``MockUpstream`` per ``type: mock`` provider, keyed by host."""
    return {
        host_key(p.base_url): MockUpstream(p)
        for p in registry.providers.values()
        if p.type == "mock"
    }


class GatewayTransport(httpx.AsyncBaseTransport):
    """Serves mock providers in-process; everything else goes to ``inner``."""

    def __init__(
        self,
        inner: Optional[httpx.AsyncBaseTransport] = None,
        mocks: Optional[dict[str, MockUpstream]] = None,
    ) -> None:
        self._inner = inner
        self.mocks: dict[str, MockUpstream] = dict(mocks or {})

    def set_mocks(self, mocks: dict[str, MockUpstream]) -> None:
        """Swap the mock set (config reload). Atomic: one attribute assignment."""
        self.mocks = dict(mocks)

    def mock_for(self, provider_name: str) -> Optional[MockUpstream]:
        return next((m for m in self.mocks.values() if m.name == provider_name), None)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        mock = self.mocks.get(host_key(request.url))
        if mock is not None:
            await request.aread()
            return await mock.handle(request)
        if self._inner is None:
            self._inner = httpx.AsyncHTTPTransport()
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        if self._inner is not None:
            await self._inner.aclose()


class OfflineGuardTransport(httpx.AsyncBaseTransport):
    """Refuses every request and records it: proof that a run stayed offline.

    Used under ``GatewayTransport`` by the demo, so a request that is not
    served by a mock provider fails (as a connection error the gateway can
    handle) instead of silently reaching the internet.
    """

    def __init__(self) -> None:
        self.blocked: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.blocked.append(str(request.url))
        raise httpx.ConnectError(f"offline mode: refused to contact {request.url.host}", request=request)
