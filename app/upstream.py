"""HTTP calls to OpenAI-compatible upstreams via httpx.

Everything here classifies transport and HTTP errors into ``UpstreamError``
(RETRY / FAILOVER / FATAL, see ``fallback``) so ``execute_with_fallback`` can
decide whether to retry the same target, move to the next one, or give up.
"""

from __future__ import annotations

import json
import time
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator, Optional

import httpx

from .fallback import FAILOVER, RETRY, UpstreamError, classify_status
from .providers import Provider


def _auth_headers(provider: Provider) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = provider.api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _is_retryable_status(status: int) -> bool:
    """Kept for callers of the old API: True when the status is worth retrying."""
    return classify_status(status) == RETRY


def parse_retry_after(headers: httpx.Headers, now: Optional[float] = None) -> Optional[float]:
    """Seconds from ``retry-after-ms`` / ``Retry-After`` (delta or HTTP date)."""
    raw_ms = headers.get("retry-after-ms")
    if raw_ms:
        try:
            return max(0.0, float(raw_ms) / 1000.0)
        except ValueError:
            pass
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    now = time.time() if now is None else now
    return max(0.0, when.timestamp() - now)


def _missing_key(provider: Provider) -> UpstreamError:
    # Deterministic misconfiguration: retrying the same target cannot help, but
    # the next provider in the chain may well have its key.
    return UpstreamError(
        503,
        f"no API key configured for provider '{provider.name}' (set {provider.api_key_env})",
        kind=FAILOVER,
        reason="no-key",
    )


def _transport_error(provider: Provider, exc: Exception) -> UpstreamError:
    if isinstance(exc, httpx.TimeoutException):
        reason = "timeout"
    elif isinstance(exc, httpx.ConnectError):
        reason = "connect-error"
    else:
        reason = "transport-error"
    return UpstreamError(504, f"{provider.name} {reason}: {exc!r}", kind=RETRY, reason=reason)


def _http_error(provider: Provider, resp: httpx.Response, text: str) -> UpstreamError:
    return UpstreamError(
        resp.status_code,
        f"{provider.name}: {text[:400]}",
        kind=classify_status(resp.status_code),
        retry_after=parse_retry_after(resp.headers),
    )


async def _post_json(
    client: httpx.AsyncClient,
    provider: Provider,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if provider.api_key() is None:
        raise _missing_key(provider)
    try:
        resp = await client.post(url, json=payload, headers=_auth_headers(provider), timeout=provider.timeout)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise _transport_error(provider, exc) from exc
    if resp.status_code >= 400:
        raise _http_error(provider, resp, resp.text)
    try:
        data = resp.json()
    except ValueError as exc:
        raise UpstreamError(502, f"{provider.name}: response is not JSON: {resp.text[:200]}", kind=RETRY) from exc
    if not isinstance(data, dict):
        raise UpstreamError(502, f"{provider.name}: unexpected response shape", kind=RETRY)
    return data


async def post_chat(
    client: httpx.AsyncClient,
    provider: Provider,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Non-streaming chat completion. Returns the parsed JSON body."""
    return await _post_json(client, provider, provider.chat_url, {**payload, "stream": False})


async def post_embeddings(
    client: httpx.AsyncClient,
    provider: Provider,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return await _post_json(client, provider, provider.embeddings_url, payload)


def streaming_body(payload: dict[str, Any]) -> dict[str, Any]:
    """Ask for a usage chunk upstream (for accounting) on top of the caller's options."""
    return {
        **payload,
        "stream": True,
        "stream_options": {**(payload.get("stream_options") or {}), "include_usage": True},
    }


async def open_chat_stream(
    client: httpx.AsyncClient,
    provider: Provider,
    payload: dict[str, Any],
) -> httpx.Response:
    """Start a streaming chat completion and return the open response.

    Raises ``UpstreamError`` if the connection fails or the upstream answers
    with an error status, i.e. *before* a single byte reaches the caller, so
    the executor can still retry or fail over. The caller must ``aclose()`` it.
    """
    if provider.api_key() is None:
        raise _missing_key(provider)
    request = client.build_request(
        "POST", provider.chat_url, json=streaming_body(payload),
        headers=_auth_headers(provider), timeout=provider.timeout,
    )
    try:
        resp = await client.send(request, stream=True)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise _transport_error(provider, exc) from exc
    if resp.status_code >= 400:
        try:
            text = (await resp.aread()).decode("utf-8", "replace")
        except (httpx.TimeoutException, httpx.TransportError):
            text = ""
        finally:
            await resp.aclose()
        raise _http_error(provider, resp, text)
    return resp


async def iter_sse_lines(resp: httpx.Response) -> AsyncIterator[bytes]:
    """Re-frame an upstream SSE body as complete events, one ``data:`` line each."""
    async for line in resp.aiter_lines():
        if line:
            yield (line + "\n\n").encode("utf-8")


async def stream_chat(
    client: httpx.AsyncClient,
    provider: Provider,
    payload: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Yield raw SSE events from an upstream streaming chat completion.

    Convenience wrapper over ``open_chat_stream`` + ``iter_sse_lines``. Errors
    before the first byte raise ``UpstreamError``; later transport errors are
    raised as ``UpstreamError`` too, but failover is no longer possible then.
    """
    resp = await open_chat_stream(client, provider, payload)
    try:
        async for event in iter_sse_lines(resp):
            yield event
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise _transport_error(provider, exc) from exc
    finally:
        await resp.aclose()


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
