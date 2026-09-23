"""HTTP calls to OpenAI-compatible upstreams via httpx.

Everything here classifies transport and HTTP errors into ``UpstreamError`` so
``fallback.execute_with_fallback`` can decide whether to retry or fail over.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from .fallback import UpstreamError
from .providers import Provider


def _auth_headers(provider: Provider) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = provider.api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _is_retryable_status(status: int) -> bool:
    # Server errors and rate limits are worth retrying/failing over; other 4xx
    # (bad request, auth, not found) are the caller's fault and will not improve.
    return status >= 500 or status == 429 or status == 408


async def post_chat(
    client: httpx.AsyncClient,
    provider: Provider,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Non-streaming chat completion. Returns the parsed JSON body."""
    if provider.api_key() is None:
        raise UpstreamError(
            503,
            f"no API key configured for provider '{provider.name}' (env {provider.api_key_env})",
            retryable=True,
        )
    try:
        resp = await client.post(
            provider.chat_url,
            json={**payload, "stream": False},
            headers=_auth_headers(provider),
            timeout=provider.timeout,
        )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise UpstreamError(504, f"{provider.name} transport error: {exc}", retryable=True) from exc

    if resp.status_code >= 400:
        raise UpstreamError(
            resp.status_code,
            f"{provider.name}: {resp.text[:400]}",
            retryable=_is_retryable_status(resp.status_code),
        )
    return resp.json()


async def post_embeddings(
    client: httpx.AsyncClient,
    provider: Provider,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if provider.api_key() is None:
        raise UpstreamError(
            503,
            f"no API key configured for provider '{provider.name}' (env {provider.api_key_env})",
            retryable=True,
        )
    try:
        resp = await client.post(
            provider.embeddings_url,
            json=payload,
            headers=_auth_headers(provider),
            timeout=provider.timeout,
        )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise UpstreamError(504, f"{provider.name} transport error: {exc}", retryable=True) from exc

    if resp.status_code >= 400:
        raise UpstreamError(
            resp.status_code,
            f"{provider.name}: {resp.text[:400]}",
            retryable=_is_retryable_status(resp.status_code),
        )
    return resp.json()


async def stream_chat(
    client: httpx.AsyncClient,
    provider: Provider,
    payload: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Yield raw SSE lines from an upstream streaming chat completion.

    Raises ``UpstreamError`` if the connection or the initial response fails
    (before any bytes are yielded), which lets the caller fail over. Once bytes
    start flowing, failover is no longer possible and errors propagate.
    """
    if provider.api_key() is None:
        raise UpstreamError(
            503,
            f"no API key configured for provider '{provider.name}' (env {provider.api_key_env})",
            retryable=True,
        )
    body = {
        **payload,
        "stream": True,
        "stream_options": {**payload.get("stream_options", {}), "include_usage": True},
    }
    try:
        async with client.stream(
            "POST",
            provider.chat_url,
            json=body,
            headers=_auth_headers(provider),
            timeout=provider.timeout,
        ) as resp:
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", "replace")
                raise UpstreamError(
                    resp.status_code,
                    f"{provider.name}: {text[:400]}",
                    retryable=_is_retryable_status(resp.status_code),
                )
            async for line in resp.aiter_lines():
                if line:
                    # Reform a complete SSE event (event lines are newline-delimited,
                    # events blank-line-delimited). Each OpenAI chunk is one data line.
                    yield (line + "\n\n").encode("utf-8")
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise UpstreamError(504, f"{provider.name} transport error: {exc}", retryable=True) from exc


def parse_sse_event(line: str) -> dict[str, Any] | None:
    """Parse one ``data:`` SSE line into its JSON object (None for anything else)."""
    if not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if data == "[DONE]" or not data:
        return None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_sse_content(line: str) -> tuple[str, dict[str, Any] | None]:
    """Extract the incremental content and any usage block from one SSE line."""
    obj = parse_sse_event(line)
    if obj is None:
        return "", None
    delta_text = ""
    for choice in obj.get("choices", []) or []:
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            delta_text += delta["content"]
    usage = obj.get("usage")
    return delta_text, usage
