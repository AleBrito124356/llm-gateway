"""FastAPI application: the OpenAI-compatible gateway surface.

Endpoints:
    POST /v1/chat/completions   - chat, streaming and non-streaming
    POST /v1/embeddings         - embeddings
    GET  /v1/models             - list gateway models
    GET  /admin/usage           - usage + cost summary (JSON or HTML)
    GET  /healthz               - liveness

Point any OpenAI SDK at this server by setting ``base_url`` to
``http://localhost:8000/v1`` and ``api_key`` to a gateway virtual key.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .accounting import Accounting, Pricing, approx_tokens
from .cache import SemanticCache
from .config import Settings
from .fallback import UpstreamError, execute_with_fallback
from .keys import KeyStore, VirtualKey
from .limits import RateLimiter, check_budget
from .providers import ProviderRegistry, Target, UnknownModelError
from .router import Router, RoutingConfig
from .schemas import (
    ChatCompletionRequest,
    EmbeddingRequest,
    build_chat_response,
)
from .upstream import parse_sse_content, post_chat, post_embeddings, stream_chat

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gateway")


class GatewayError(Exception):
    """A request-scoped error that maps to an OpenAI-style error response."""

    def __init__(self, status_code: int, message: str, err_type: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type


def openai_error(status_code: int, message: str, err_type: str, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "code": status_code}},
        headers=headers or {},
    )


def make_embedder(client: httpx.AsyncClient, registry: ProviderRegistry, embed_model: str):
    async def embed(text: str) -> list[float]:
        target = registry.resolve(embed_model)
        payload = {"model": target.upstream_model, "input": text}
        data = await post_embeddings(client, target.provider, payload)
        return data["data"][0]["embedding"]

    return embed


def build_app(settings: Optional[Settings] = None) -> FastAPI:
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
        embedder = make_embedder(client, registry, settings.embed_model)
        cache = SemanticCache(
            db,
            embedder,
            ttl_seconds=settings.cache_ttl_seconds,
            similarity_threshold=settings.cache_similarity_threshold,
            semantic_enabled=settings.semantic_cache_enabled,
        )
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

    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry
    app.state.router = router
    app.state.keystore = keystore
    app.state.accounting = accounting
    app.state.limiter = limiter

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
            raise GatewayError(429, f"rate limit exceeded for key '{key.name}'", "rate_limit_error")
        spent = accounting.month_cost(key.name)
        budget = check_budget(spent, key.monthly_budget_usd)
        headers.update(budget.headers())
        if not budget.allowed:
            raise GatewayError(
                402,
                f"monthly budget of ${key.monthly_budget_usd:.2f} exhausted for key '{key.name}'",
                "budget_exceeded",
            )
        return headers

    def caching_allowed(route, body: dict[str, Any]) -> tuple[bool, bool]:
        exact = settings.cache_enabled and route.cache and body.get("cache") is not False
        semantic = (
            exact
            and settings.semantic_cache_enabled
            and route.semantic_cache
            and not body.get("tools")
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
            req = ChatCompletionRequest(**body)
        except GatewayError as err:
            headers = err_headers(err)
            return openai_error(err.status_code, err.message, err.err_type, headers)
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

        # Cache lookup.
        if exact_ok:
            try:
                hit = await cache.lookup(gateway_model, body, allow_semantic=semantic_ok)
            except Exception as exc:  # a failing embedder must not break the request
                logger.warning("cache lookup failed: %s", exc)
                hit = None
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
                        replay_stream(hit.response, gateway_model),
                        media_type="text/event-stream", headers=headers,
                    )
                return JSONResponse(content=hit.response, headers=headers)

        chain = router.resolve_chain(gateway_model)
        base_headers = {**limit_headers, "X-Cache": "MISS", "X-Gateway-Model": gateway_model}
        if decision.get("reason") not in (None, "exact"):
            base_headers["X-Gateway-Route"] = decision.get("reason", "")

        if req.stream:
            return StreamingResponse(
                stream_and_account(
                    app, client, chain, body, gateway_model, key, exact_ok, semantic_ok,
                ),
                media_type="text/event-stream", headers=base_headers,
            )

        async def attempt(target: Target) -> dict[str, Any]:
            return await post_chat(client, target.provider, {**body, "model": target.upstream_model})

        try:
            result = await execute_with_fallback(
                chain, attempt,
                max_retries_per_target=settings.max_retries_per_target,
                backoff_base=settings.backoff_base_seconds,
                backoff_cap=settings.backoff_cap_seconds,
                target_name=lambda t: f"{t.provider.name}:{t.upstream_model}",
            )
        except UpstreamError as err:
            status = 502 if err.status_code >= 500 else err.status_code
            return openai_error(status, err.message, "upstream_error", base_headers)

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
            try:
                await cache.store(gateway_model, body, data, allow_semantic=semantic_ok)
            except Exception as exc:
                logger.warning("cache store failed: %s", exc)

        headers = {**base_headers, "X-Gateway-Provider": used.provider.name, "X-Upstream-Model": used.upstream_model}
        return JSONResponse(content=data, headers=headers)

    # ---------------------------------------------------------- embeddings
    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        try:
            key = authenticate(request)
            limit_headers = enforce_limits(key)
            body = await request.json()
            req = EmbeddingRequest(**body)
        except GatewayError as err:
            return openai_error(err.status_code, err.message, err.err_type, err_headers(err))
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

        async def attempt(target: Target) -> dict[str, Any]:
            return await post_embeddings(client, target.provider, {**body, "model": target.upstream_model})

        try:
            result = await execute_with_fallback(
                chain, attempt,
                max_retries_per_target=settings.max_retries_per_target,
                backoff_base=settings.backoff_base_seconds,
                backoff_cap=settings.backoff_cap_seconds,
                target_name=lambda t: f"{t.provider.name}:{t.upstream_model}",
            )
        except UpstreamError as err:
            status = 502 if err.status_code >= 500 else err.status_code
            return openai_error(status, err.message, "upstream_error", limit_headers)

        data = result.value
        used: Target = result.target
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or approx_tokens(_embed_input_text(req)))
        accounting.record(
            virtual_key=key.name, model=model, provider=used.provider.name,
            endpoint="embeddings", prompt_tokens=prompt_tokens, completion_tokens=0, cached=False,
        )
        return JSONResponse(content=data, headers={**limit_headers, "X-Gateway-Provider": used.provider.name})

    # -------------------------------------------------------------- models
    @app.get("/v1/models")
    async def list_models():
        data = [{"id": m, "object": "model", "owned_by": "gateway"} for m in registry.list_models()]
        data.append({"id": "auto", "object": "model", "owned_by": "gateway"})
        for alias in routing.aliases:
            data.append({"id": alias, "object": "model", "owned_by": "gateway"})
        return {"object": "list", "data": data}

    # --------------------------------------------------------------- admin
    @app.get("/admin/usage")
    async def admin_usage(request: Request):
        admin_token = os.getenv("ADMIN_TOKEN")
        if admin_token and request.headers.get("x-admin-token") != admin_token:
            return openai_error(401, "invalid admin token", "authentication_error")
        summary = accounting.summary()
        want_json = request.query_params.get("format") == "json" or "application/json" in request.headers.get("accept", "")
        if want_json:
            return JSONResponse(summary)
        return HTMLResponse(render_usage_html(summary))

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "providers": len(registry.providers), "models": len(registry.models)}

    @app.get("/")
    async def root():
        return {
            "name": "llm-gateway",
            "docs": "/docs",
            "endpoints": ["/v1/chat/completions", "/v1/embeddings", "/v1/models", "/admin/usage"],
        }

    return app


def err_headers(err: GatewayError) -> dict[str, str]:
    if err.status_code == 429:
        return {"Retry-After": "1"}
    return {}


# ------------------------------------------------------------- helpers


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


async def replay_stream(response: dict[str, Any], model: str):
    """Turn a cached full completion back into a minimal SSE stream."""
    content = _completion_text(response)
    completion_id = response.get("id", "chatcmpl-cache")
    first = {
        "id": completion_id, "object": "chat.completion.chunk", "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n".encode("utf-8")
    body = {
        "id": completion_id, "object": "chat.completion.chunk", "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(body)}\n\n".encode("utf-8")
    last = {
        "id": completion_id, "object": "chat.completion.chunk", "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(last)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


async def stream_and_account(
    app: FastAPI,
    client: httpx.AsyncClient,
    chain: list[Target],
    body: dict[str, Any],
    gateway_model: str,
    key: VirtualKey,
    exact_ok: bool,
    semantic_ok: bool,
):
    """Stream from the first working target, then record usage and cache."""
    accounting: Accounting = app.state.accounting
    cache: SemanticCache = app.state.cache
    accumulated: list[str] = []
    usage: dict[str, Any] = {}
    used: Optional[Target] = None
    started = False
    last_error: Optional[UpstreamError] = None

    for target in chain:
        payload = {**body, "model": target.upstream_model}
        try:
            async for chunk in stream_chat(client, target.provider, payload):
                started = True
                used = target
                text, u = parse_sse_content(chunk.decode("utf-8", "replace").strip())
                if text:
                    accumulated.append(text)
                if u:
                    usage = u
                yield chunk
            used = used or target
            break
        except UpstreamError as err:
            last_error = err
            if started or not err.retryable:
                yield _sse_error(err)
                return
            continue
    else:
        if last_error is not None:
            yield _sse_error(last_error)
        return

    content = "".join(accumulated)
    prompt_tokens = int(usage.get("prompt_tokens") or approx_tokens(
        "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else "" for m in body.get("messages", []))
    ))
    completion_tokens = int(usage.get("completion_tokens") or approx_tokens(content))
    provider_name = used.provider.name if used else (chain[0].provider.name if chain else "unknown")
    accounting.record(
        virtual_key=key.name, model=gateway_model, provider=provider_name,
        endpoint="chat", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cached=False,
    )
    if exact_ok and content:
        response = build_chat_response(
            gateway_model, content,
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens},
        )
        try:
            await cache.store(gateway_model, body, response, allow_semantic=semantic_ok)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache store failed: %s", exc)


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
