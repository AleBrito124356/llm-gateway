"""FastAPI application: the OpenAI-compatible gateway surface.

Endpoints:
    POST /v1/chat/completions   - chat, streaming and non-streaming
    POST /v1/embeddings         - embeddings
    GET  /v1/models             - list gateway models
    GET  /admin/usage           - usage + cost summary (JSON or HTML)
    GET  /admin/health          - circuit-breaker state and last error per target
    GET  /healthz               - liveness

Point any OpenAI SDK at this server by setting ``base_url`` to
``http://localhost:8000/v1`` and ``api_key`` to a gateway virtual key.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .accounting import Accounting, Pricing, approx_tokens, next_month_start_epoch
from .cache import CacheProbe, SemanticCache
from .config import Settings
from .fallback import CircuitBreaker, FallbackResult, UpstreamError, execute_with_fallback, summarize_attempts
from .keys import KeyStore, VirtualKey
from .limits import RateLimiter, check_budget, retry_after_header
from .providers import ProviderRegistry, Target, UnknownModelError
from .router import Router, RoutingConfig
from .schemas import (
    ChatCompletionRequest,
    EmbeddingRequest,
    build_chat_response,
)
from .upstream import iter_sse_lines, open_chat_stream, parse_sse_event, post_chat, post_embeddings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gateway")

# Request fields that configure the gateway itself. They are consumed here and
# never forwarded: OpenAI and most compatible servers reject unknown arguments.
GATEWAY_ONLY_FIELDS = frozenset({"cache"})


class GatewayError(Exception):
    """A request-scoped error that maps to an OpenAI-style error response."""

    def __init__(
        self,
        status_code: int,
        message: str,
        err_type: str = "invalid_request_error",
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type
        self.headers = headers or {}


def openai_error(status_code: int, message: str, err_type: str, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "code": status_code}},
        headers=headers or {},
    )


def upstream_body(body: dict[str, Any]) -> dict[str, Any]:
    """The caller's body minus gateway-only fields."""
    return {k: v for k, v in body.items() if k not in GATEWAY_ONLY_FIELDS}


def client_status(err: UpstreamError) -> int:
    """HTTP status the caller sees for an upstream failure.

    The executor already turns exhausted chains into 502 (or 503 when every
    circuit is open) and passes caller errors (400/413/422) through. Upstream
    auth failures are the gateway's problem, not the caller's: surfacing them
    as 401/403 would make the OpenAI SDK report the caller's own gateway key
    as invalid, so they are mapped to 502 defensively.
    """
    if err.status_code in (401, 403) or (err.status_code >= 500 and err.status_code != 503):
        return 502
    return err.status_code


def upstream_error_response(err: UpstreamError, headers: dict[str, str]) -> JSONResponse:
    headers = dict(headers)
    if err.attempts:
        headers["X-Gateway-Attempts"] = summarize_attempts(err.attempts)
    status = client_status(err)
    if status == 503 and err.retry_after is not None:
        headers["Retry-After"] = str(retry_after_header(err.retry_after))
    return openai_error(status, err.message, "upstream_error", headers)


def served_headers(result: FallbackResult) -> dict[str, str]:
    target: Target = result.target
    return {
        "X-Gateway-Provider": target.provider.name,
        "X-Upstream-Model": target.upstream_model,
        "X-Gateway-Attempts": result.summary,
    }


def make_embedder(client: httpx.AsyncClient, registry: ProviderRegistry, embed_model: str):
    async def embed(text: str) -> list[float]:
        target = registry.resolve(embed_model)
        data = await post_embeddings(client, target.provider, target.payload({"input": text}))
        return data["data"][0]["embedding"]

    return embed


def build_app(settings: Optional[Settings] = None, transport: Optional[httpx.AsyncBaseTransport] = None) -> FastAPI:
    """Build the gateway app.

    ``transport`` replaces the network layer of the shared upstream client (and
    of the cache embedder, which uses the same client). Tests pass an
    ``httpx.MockTransport`` here; production leaves it ``None``.
    """
    settings = settings or Settings.from_env()

    from .db import Database

    db = Database(settings.db_path)
    registry = ProviderRegistry.from_file(settings.providers_file)
    routing = RoutingConfig.from_file(settings.routing_file)
    router = Router(routing, registry)
    keystore = KeyStore.from_file(settings.keys_file)
    pricing = Pricing.from_file(settings.pricing_file)
    accounting = Accounting(db, pricing)
    limiter = RateLimiter()
    breaker = CircuitBreaker(settings.breaker_failure_threshold, settings.breaker_cooldown_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = httpx.AsyncClient(timeout=settings.request_timeout_seconds, transport=transport)
        embedder = make_embedder(client, registry, settings.embed_model)
        cache = SemanticCache(
            db,
            embedder,
            ttl_seconds=settings.cache_ttl_seconds,
            similarity_threshold=settings.cache_similarity_threshold,
            semantic_enabled=settings.semantic_cache_enabled,
            max_entries=settings.cache_max_entries,
            maintenance_interval_seconds=settings.cache_purge_interval_seconds,
        )
        cache.maintain()
        app.state.client = client
        app.state.cache = cache
        logger.info(
            "gateway ready: %d providers, %d models, %d keys",
            len(registry.providers), len(registry.models), len(keystore),
        )
        try:
            yield
        finally:
            await client.aclose()
            db.close()

    app = FastAPI(title="llm-gateway", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry
    app.state.router = router
    app.state.keystore = keystore
    app.state.accounting = accounting
    app.state.limiter = limiter
    app.state.breaker = breaker

    async def run_chain(chain: list[Target], attempt) -> FallbackResult:
        return await execute_with_fallback(
            chain, attempt,
            max_retries_per_target=settings.max_retries_per_target,
            backoff_base=settings.backoff_base_seconds,
            backoff_cap=settings.backoff_cap_seconds,
            target_name=lambda t: t.name,
            breaker=breaker,
        )

    # ------------------------------------------------------------------ auth
    def authenticate(request: Request) -> VirtualKey:
        if not settings.require_auth:
            return VirtualKey(name="anonymous", key_hash="", rpm=6000, monthly_budget_usd=None)
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else header.strip()
        key = keystore.authenticate(token)
        if key is None:
            raise GatewayError(401, "invalid gateway API key", "authentication_error")
        return key

    def enforce_limits(key: VirtualKey) -> dict[str, str]:
        rate = limiter.check(key.name, key.rpm)
        headers = rate.headers()
        if not rate.allowed:
            raise GatewayError(
                429, f"rate limit exceeded for key '{key.name}'", "rate_limit_error", headers,
            )
        spent = accounting.month_cost(key.name)
        budget = check_budget(spent, key.monthly_budget_usd, resets_at=next_month_start_epoch())
        headers.update(budget.headers())
        if not budget.allowed:
            raise GatewayError(
                402,
                f"monthly budget of ${key.monthly_budget_usd:.2f} exhausted for key '{key.name}'",
                "budget_exceeded",
                headers,
            )
        return headers

    def require_admin(request: Request) -> None:
        token = settings.admin_token
        if token and not hmac.compare_digest(
            request.headers.get("x-admin-token", "").encode("utf-8"), token.encode("utf-8")
        ):
            raise GatewayError(401, "invalid admin token", "authentication_error")

    def caching_allowed(route, body: dict[str, Any]) -> tuple[bool, bool]:
        exact = settings.cache_enabled and route.cache and body.get("cache") is not False
        semantic = (
            exact
            and settings.semantic_cache_enabled
            and route.semantic_cache
            and not body.get("tools")
            and not body.get("functions")
            and (body.get("n") in (None, 1))
        )
        return exact, semantic

    # -------------------------------------------------------------- chat
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            key = authenticate(request)
            limit_headers = enforce_limits(key)
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("the request body must be a JSON object")
            req = ChatCompletionRequest(**body)
        except GatewayError as err:
            return openai_error(err.status_code, err.message, err.err_type, err.headers)
        except (json.JSONDecodeError, ValueError, TypeError) as err:
            return openai_error(400, f"invalid request body: {err}", "invalid_request_error")

        # Routing.
        gateway_model, decision = router.resolve_model(req.model, req.messages, req.wants_json())
        try:
            route = registry.route(gateway_model)
        except UnknownModelError:
            return openai_error(404, f"model '{gateway_model}' is not configured", "model_not_found", limit_headers)
        if not key.allows_model(gateway_model):
            return openai_error(403, f"key '{key.name}' is not allowed to use model '{gateway_model}'", "permission_error", limit_headers)

        cache: SemanticCache = app.state.cache
        client: httpx.AsyncClient = app.state.client
        exact_ok, semantic_ok = caching_allowed(route, body)
        wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        # Cache lookup.
        probe = CacheProbe(None)
        if exact_ok:
            try:
                probe = await cache.probe(gateway_model, body, allow_semantic=semantic_ok)
            except Exception as exc:  # noqa: BLE001 - the cache must never break a request
                logger.warning("cache lookup failed: %s", exc)
            hit = probe.hit
            if hit is not None:
                usage = hit.response.get("usage", {}) or {}
                accounting.record(
                    virtual_key=key.name, model=gateway_model, provider=route.provider,
                    endpoint="chat", prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)), cached=True,
                )
                headers = {
                    **limit_headers,
                    "X-Cache": f"HIT;{hit.kind}",
                    "X-Cache-Similarity": f"{hit.similarity:.4f}",
                    "X-Gateway-Model": gateway_model,
                }
                if req.stream:
                    return StreamingResponse(
                        replay_stream(hit.response, gateway_model, include_usage=wants_usage),
                        media_type="text/event-stream", headers=headers,
                    )
                return JSONResponse(content=hit.response, headers=headers)

        chain = router.resolve_chain(gateway_model)
        base_headers = {**limit_headers, "X-Cache": "MISS", "X-Gateway-Model": gateway_model}
        if decision.get("reason") not in (None, "exact"):
            base_headers["X-Gateway-Route"] = decision.get("reason", "")
        forward = upstream_body(body)

        if req.stream:
            # Open the upstream stream (with retries and failover) *before*
            # answering, so a dead chain is a proper HTTP error and the headers
            # can name the provider that is actually streaming.
            async def open_attempt(target: Target) -> httpx.Response:
                return await open_chat_stream(client, target.provider, target.payload(forward))

            try:
                opened = await run_chain(chain, open_attempt)
            except UpstreamError as err:
                return upstream_error_response(err, base_headers)
            return StreamingResponse(
                relay_stream(
                    app, opened.value, opened.target, body, gateway_model, key,
                    exact_ok, semantic_ok, probe, wants_usage,
                ),
                media_type="text/event-stream", headers={**base_headers, **served_headers(opened)},
            )

        async def attempt(target: Target) -> dict[str, Any]:
            return await post_chat(client, target.provider, target.payload(forward))

        try:
            result = await run_chain(chain, attempt)
        except UpstreamError as err:
            return upstream_error_response(err, base_headers)

        data = result.value
        used: Target = result.target
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or approx_tokens(_prompt_text(req)))
        completion_tokens = int(usage.get("completion_tokens") or approx_tokens(_completion_text(data)))
        accounting.record(
            virtual_key=key.name, model=gateway_model, provider=used.provider.name,
            endpoint="chat", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cached=False,
        )
        if exact_ok:
            await _store_in_cache(cache, gateway_model, body, data, semantic_ok, probe)

        return JSONResponse(content=data, headers={**base_headers, **served_headers(result)})

    # ---------------------------------------------------------- embeddings
    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        try:
            key = authenticate(request)
            limit_headers = enforce_limits(key)
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("the request body must be a JSON object")
            req = EmbeddingRequest(**body)
        except GatewayError as err:
            return openai_error(err.status_code, err.message, err.err_type, err.headers)
        except (json.JSONDecodeError, ValueError, TypeError) as err:
            return openai_error(400, f"invalid request body: {err}", "invalid_request_error")

        model = routing.aliases.get(req.model, req.model)
        try:
            registry.route(model)
        except UnknownModelError:
            return openai_error(404, f"model '{model}' is not configured", "model_not_found", limit_headers)
        if not key.allows_model(model):
            return openai_error(403, f"key '{key.name}' is not allowed to use model '{model}'", "permission_error", limit_headers)

        client: httpx.AsyncClient = app.state.client
        chain = router.resolve_chain(model)
        forward = upstream_body(body)

        async def attempt(target: Target) -> dict[str, Any]:
            return await post_embeddings(client, target.provider, target.payload(forward))

        try:
            result = await run_chain(chain, attempt)
        except UpstreamError as err:
            return upstream_error_response(err, {**limit_headers, "X-Gateway-Model": model})

        data = result.value
        used: Target = result.target
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or approx_tokens(_embed_input_text(req)))
        accounting.record(
            virtual_key=key.name, model=model, provider=used.provider.name,
            endpoint="embeddings", prompt_tokens=prompt_tokens, completion_tokens=0, cached=False,
        )
        # X-Gateway-Model names the model that produced the vectors: after a
        # failover it differs from the requested one, and so may the dimensions.
        return JSONResponse(content=data, headers={
            **limit_headers, "X-Gateway-Model": used.gateway_model, **served_headers(result),
        })

    # -------------------------------------------------------------- models
    @app.get("/v1/models")
    async def list_models():
        data = [{"id": m, "object": "model", "created": 0, "owned_by": "gateway"} for m in registry.list_models()]
        data.append({"id": "auto", "object": "model", "created": 0, "owned_by": "gateway"})
        for alias in routing.aliases:
            data.append({"id": alias, "object": "model", "created": 0, "owned_by": "gateway"})
        return {"object": "list", "data": data}

    # --------------------------------------------------------------- admin
    @app.get("/admin/usage")
    async def admin_usage(request: Request):
        try:
            require_admin(request)
        except GatewayError as err:
            return openai_error(err.status_code, err.message, err.err_type)
        summary = accounting.summary()
        want_json = request.query_params.get("format") == "json" or "application/json" in request.headers.get("accept", "")
        if want_json:
            return JSONResponse(summary)
        return HTMLResponse(render_usage_html(summary))

    @app.get("/admin/health")
    async def admin_health(request: Request):
        try:
            require_admin(request)
        except GatewayError as err:
            return openai_error(err.status_code, err.message, err.err_type)
        targets: dict[str, dict[str, Any]] = {}
        for model in registry.list_models():
            target = registry.resolve(model)
            targets.setdefault(target.name, {"state": "closed", "consecutive_failures": 0,
                                             "successes": 0, "failures": 0, "last_status": None,
                                             "last_error": None, "last_error_at": None,
                                             "last_success_at": None, "retry_in_seconds": 0.0})
        targets.update(breaker.snapshot())
        unhealthy = sorted(name for name, t in targets.items() if t["state"] != "closed")
        return {
            "status": "degraded" if unhealthy else "ok",
            "unhealthy_targets": unhealthy,
            "breaker": {
                "enabled": breaker.enabled,
                "failure_threshold": breaker.failure_threshold,
                "cooldown_seconds": breaker.cooldown_seconds,
            },
            "providers": {
                name: {"base_url": p.base_url, "api_key_set": p.api_key() is not None}
                for name, p in registry.providers.items()
            },
            "targets": targets,
        }

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "providers": len(registry.providers), "models": len(registry.models)}

    @app.get("/")
    async def root():
        return {
            "name": "llm-gateway",
            "docs": "/docs",
            "endpoints": ["/v1/chat/completions", "/v1/embeddings", "/v1/models", "/admin/usage",
                          "/admin/health"],
        }

    return app


# ------------------------------------------------------------- helpers


async def _store_in_cache(
    cache: SemanticCache,
    model: str,
    body: dict[str, Any],
    response: dict[str, Any],
    semantic_ok: bool,
    probe: CacheProbe,
) -> None:
    try:
        await cache.store(
            model, body, response, allow_semantic=semantic_ok,
            vector=probe.vector, embed=not probe.embed_failed,
        )
    except Exception as exc:  # noqa: BLE001 - caching is best effort
        logger.warning("cache store failed: %s", exc)


def _prompt_text(req: ChatCompletionRequest) -> str:
    return "\n".join(m.text() for m in req.messages)


def _completion_text(data: dict[str, Any]) -> str:
    parts = []
    for choice in data.get("choices", []) or []:
        message = choice.get("message") or {}
        if isinstance(message.get("content"), str):
            parts.append(message["content"])
    return "\n".join(parts)


def _embed_input_text(req: EmbeddingRequest) -> str:
    if isinstance(req.input, str):
        return req.input
    if isinstance(req.input, list):
        return " ".join(str(x) for x in req.input)
    return str(req.input)


def _sse(obj: Any) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


async def replay_stream(response: dict[str, Any], model: str, include_usage: bool = False):
    """Turn a cached full completion back into an OpenAI-shaped SSE stream."""
    completion_id = response.get("id", "chatcmpl-cache")
    created = int(response.get("created") or time.time())

    def chunk(choices: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        return {
            "id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": choices, **extra,
        }

    for choice in response.get("choices") or [{"index": 0, "message": {}}]:
        index = choice.get("index", 0)
        message = choice.get("message") or {}
        yield _sse(chunk([{"index": index, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
        if isinstance(message.get("content"), str) and message["content"]:
            yield _sse(chunk([{"index": index, "delta": {"content": message["content"]}, "finish_reason": None}]))
        if message.get("tool_calls"):
            calls = [{**call, "index": i} for i, call in enumerate(message["tool_calls"])]
            yield _sse(chunk([{"index": index, "delta": {"tool_calls": calls}, "finish_reason": None}]))
        finish = choice.get("finish_reason") or "stop"
        yield _sse(chunk([{"index": index, "delta": {}, "finish_reason": finish}]))
    if include_usage:
        yield _sse(chunk([], usage=response.get("usage") or {}))
    yield b"data: [DONE]\n\n"


async def relay_stream(
    app: FastAPI,
    resp: httpx.Response,
    target: Target,
    body: dict[str, Any],
    gateway_model: str,
    key: VirtualKey,
    exact_ok: bool,
    semantic_ok: bool,
    probe: CacheProbe,
    wants_usage: bool,
):
    """Relay an open upstream stream, then record usage and cache the answer.

    The gateway always asks the upstream for a usage chunk (for accounting),
    but only forwards it when the caller asked for one: the chunk has an empty
    ``choices`` list, which clients that did not opt in do not expect. Usage is
    recorded even if the client disconnects mid-stream, because the upstream
    tokens were spent either way; only complete answers are cached.
    """
    accounting: Accounting = app.state.accounting
    cache: SemanticCache = app.state.cache
    breaker: CircuitBreaker = app.state.breaker
    accumulated: list[str] = []
    usage: dict[str, Any] = {}
    saw_tool_calls = False
    finish_reason: Optional[str] = None
    completed = False
    try:
        try:
            async for chunk in iter_sse_lines(resp):
                event = parse_sse_event(chunk.decode("utf-8", "replace").strip())
                if event is not None:
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices")
                    for choice in choices or []:
                        if choice.get("index", 0) != 0:
                            continue
                        delta = choice.get("delta") or {}
                        if isinstance(delta.get("content"), str):
                            accumulated.append(delta["content"])
                        if delta.get("tool_calls"):
                            saw_tool_calls = True
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                    if choices == [] and "usage" in event and not wants_usage:
                        continue
                yield chunk
            completed = True
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            err = UpstreamError(504, f"{target.provider.name} stream interrupted: {exc!r}")
            breaker.record_failure(target.name, err.status_code, err.message)
            logger.warning("stream from %s interrupted: %r", target.name, exc)
            yield _sse_error(err)
    finally:
        content = "".join(accumulated)
        prompt_tokens = int(usage.get("prompt_tokens") or approx_tokens(
            "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else ""
                      for m in body.get("messages", []))
        ))
        completion_tokens = int(usage.get("completion_tokens") or approx_tokens(content))
        accounting.record(
            virtual_key=key.name, model=gateway_model, provider=target.provider.name,
            endpoint="chat", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cached=False,
        )
        await resp.aclose()

    if completed and exact_ok and content and not saw_tool_calls and body.get("n") in (None, 1):
        response = build_chat_response(
            gateway_model, content,
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens},
            finish_reason=finish_reason or "stop",
        )
        await _store_in_cache(cache, gateway_model, body, response, semantic_ok, probe)


def _sse_error(err: UpstreamError) -> bytes:
    payload = {"error": {"message": err.message, "type": "upstream_error", "code": err.status_code}}
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode("utf-8")


def render_usage_html(summary: dict[str, Any]) -> str:
    totals = summary.get("totals", {})

    def rows(items: list[dict], columns: list[tuple[str, str]]) -> str:
        out = []
        for item in items:
            cells = "".join(f"<td>{_fmt(item.get(col))}</td>" for col, _ in columns)
            out.append(f"<tr>{cells}</tr>")
        return "".join(out)

    key_cols = [("virtual_key", "Key"), ("requests", "Requests"), ("total_tokens", "Tokens"),
                ("cache_hits", "Cache hits"), ("cost_usd", "Cost USD")]
    model_cols = [("model", "Model"), ("provider", "Provider"), ("requests", "Requests"),
                  ("total_tokens", "Tokens"), ("cache_hits", "Cache hits"), ("cost_usd", "Cost USD")]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>llm-gateway usage</title>
<style>
 body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:2rem;color:#18181b;background:#fafafa}}
 h1{{font-size:1.4rem}} h2{{font-size:1.05rem;margin-top:2rem;color:#3f3f46}}
 table{{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.06);border-radius:8px;overflow:hidden}}
 th,td{{padding:.55rem .8rem;text-align:left;border-bottom:1px solid #f1f1f4;font-size:.9rem}}
 th{{background:#f4f4f5;font-weight:600}} .kpi{{display:flex;gap:1rem;flex-wrap:wrap}}
 .card{{background:#fff;border:1px solid #ececf0;border-radius:10px;padding:1rem 1.2rem;min-width:140px}}
 .card b{{display:block;font-size:1.5rem;color:#2563eb}} .card span{{color:#71717a;font-size:.8rem}}
</style></head><body>
<h1>llm-gateway usage</h1>
<div class="kpi">
 <div class="card"><b>{_fmt(totals.get('requests'))}</b><span>requests</span></div>
 <div class="card"><b>{_fmt(totals.get('total_tokens'))}</b><span>tokens</span></div>
 <div class="card"><b>{_fmt(totals.get('cache_hits'))}</b><span>cache hits</span></div>
 <div class="card"><b>${_fmt(totals.get('cost_usd'))}</b><span>upstream cost</span></div>
</div>
<h2>By key</h2>
<table><thead><tr>{''.join(f'<th>{label}</th>' for _, label in key_cols)}</tr></thead>
<tbody>{rows(summary.get('by_key', []), key_cols)}</tbody></table>
<h2>By model</h2>
<table><thead><tr>{''.join(f'<th>{label}</th>' for _, label in model_cols)}</tr></thead>
<tbody>{rows(summary.get('by_model', []), model_cols)}</tbody></table>
</body></html>"""


def _fmt(value: Any) -> str:
    if value is None:
        return "0"
    if isinstance(value, float):
        return f"{value:,.4f}" if value < 1 else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# Module-level app for ``uvicorn app.main:app``.
app = build_app()
