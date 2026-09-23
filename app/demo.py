"""``python -m app demo``: an offline, scripted tour of every gateway feature.

The real FastAPI app is booted in-process against ``examples/demo`` (all
providers are ``type: mock``) and driven over ASGI, so nothing listens on a
port. The upstream client sits on an ``OfflineGuardTransport``: any request
that is not answered by a mock provider is refused and counted, and the demo
fails if that count is not zero. No API key, no network, no cost.

Every stage checks what it shows (status codes, X-Cache values, routed model,
failover hops, limit headers), so the demo doubles as an end-to-end smoke test:
the exit code is non-zero if any expectation is not met.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .config import Settings

DEMO_KEY = "sk-gw-demo-000-not-a-real-secret"
BURST_KEY = "sk-gw-demo-burst-not-a-real-secret"
TINY_BUDGET_KEY = "sk-gw-demo-tiny-budget-not-a-real-secret"
CHEAP_ONLY_KEY = "sk-gw-demo-cheap-only-not-a-real-secret"

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "examples" / "demo"

SHOWN_HEADERS = (
    "x-cache", "x-cache-similarity", "x-gateway-model", "x-gateway-route", "x-gateway-provider",
    "x-gateway-attempts",
)
RATE_HEADERS = ("x-ratelimit-limit", "x-ratelimit-remaining", "retry-after")
BUDGET_HEADERS = ("x-budget-limit", "x-budget-spent", "x-budget-remaining", "retry-after")


@dataclass
class DemoReport:
    checks: list[tuple[str, bool]] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for _, ok in self.checks if ok)

    @property
    def ok(self) -> bool:
        return self.passed == len(self.checks) and not self.blocked


def demo_settings(config_dir: Path, db_path: str) -> Settings:
    """Settings for the scripted run: independent of the caller's environment."""
    return Settings.from_env(
        env={},
        config_dir=str(config_dir),
        db_path=db_path,
        embed_model="nvidia/nv-embedqa-e5-v5",
        cache_similarity_threshold=0.85,
        max_retries_per_target=1,
        backoff_base_seconds=0.05,
        backoff_cap_seconds=0.5,
        breaker_failure_threshold=2,
        breaker_cooldown_seconds=30,
    )


class _Runner:
    def __init__(self, client: httpx.AsyncClient, out: Callable[[str], None], report: DemoReport) -> None:
        self.client = client
        self.out = out
        self.report = report

    def stage(self, number: int, title: str) -> None:
        self.out("")
        self.out(f"[{number}] {title}")

    def note(self, text: str) -> None:
        self.out(f"    {text}")

    def check(self, condition: bool, description: str) -> bool:
        self.report.checks.append((description, bool(condition)))
        self.out(f"      {'ok  ' if condition else 'FAIL'} {description}")
        return bool(condition)

    async def chat(
        self,
        label: str,
        content: Any,
        *,
        model: str = "auto",
        key: str = DEMO_KEY,
        show: tuple[str, ...] = SHOWN_HEADERS,
        **extra: Any,
    ) -> httpx.Response:
        messages = content if isinstance(content, list) else [{"role": "user", "content": content}]
        return await self.post("/v1/chat/completions", label, {"model": model, "messages": messages, **extra},
                               key=key, show=show)

    async def post(self, path: str, label: str, payload: dict[str, Any], *, key: str = DEMO_KEY,
                   show: tuple[str, ...] = SHOWN_HEADERS) -> httpx.Response:
        started = time.perf_counter()
        response = await self.client.post(path, json=payload, headers={"Authorization": f"Bearer {key}"})
        elapsed = (time.perf_counter() - started) * 1000
        self.out(f"    > {label}")
        self.out(f"      {response.status_code} in {elapsed:.0f} ms")
        for name in show:
            if name in response.headers:
                self.out(f"        {name}: {response.headers[name]}")
        return response


def _content(response: httpx.Response) -> str:
    try:
        return response.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def _sse(response: httpx.Response) -> list[Any]:
    events: list[Any] = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


async def _stages(r: _Runner) -> None:
    # 1 -------------------------------------------------------------- routing
    r.stage(1, "Auto routing: the virtual 'auto' model picks cheap vs strong per request")
    cheap = await r.chat('auto: "What is 2 + 2?"', "What is 2 + 2?")
    r.check(cheap.headers.get("x-gateway-model") == "meta/llama-3.1-8b-instruct", "plain question -> cheap 8B model")
    code = await r.chat('auto: a Python snippet', "Why does this fail?\n```python\ndef add(a, b):\n    return a + c\n```")
    r.check(code.headers.get("x-gateway-route") == "code"
            and code.headers.get("x-gateway-model") == "meta/llama-3.3-70b-instruct", "code -> strong 70B model")
    js = await r.chat("auto: JSON mode", "List three primary colors.", response_format={"type": "json_object"})
    r.check(js.headers.get("x-gateway-route") == "json_mode" and json.loads(_content(js)).get("mock") is True,
            "JSON mode -> strong model, and the answer is valid JSON")
    alias = await r.chat('model "gpt-4" (an alias)', "Hello from existing OpenAI code", model="gpt-4")
    r.check(alias.headers.get("x-gateway-model") == "meta/llama-3.3-70b-instruct", "gpt-4 alias -> Llama 70B")

    # 2 ---------------------------------------------------------------- cache
    r.stage(2, "Two-layer cache: exact hash, then semantic similarity within the same context")
    q = "What is the capital of France?"
    first = await r.chat(f'"{q}"', q, model="meta/llama-3.3-70b-instruct")
    r.check(first.headers.get("x-cache") == "MISS", "first ask is a MISS and goes upstream")
    again = await r.chat("the identical request again", q, model="meta/llama-3.3-70b-instruct")
    r.check(again.headers.get("x-cache") == "HIT;EXACT", "identical request -> HIT;EXACT, $0")
    para = await r.chat('"Tell me the capital of France, please."', "Tell me the capital of France, please.",
                        model="meta/llama-3.3-70b-instruct")
    r.check(para.headers.get("x-cache") == "HIT;SEMANTIC" and _content(para) == _content(first),
            "a reworded question -> HIT;SEMANTIC with the cached answer")
    spain = await r.chat('"What is the capital of Spain?"', "What is the capital of Spain?",
                         model="meta/llama-3.3-70b-instruct")
    r.check(spain.headers.get("x-cache") == "MISS", "a different question is not matched")
    conv_a = [{"role": "user", "content": "Tell me about Python lists"},
              {"role": "assistant", "content": "Lists are ordered and mutable."},
              {"role": "user", "content": "Give me an example"}]
    conv_b = [{"role": "user", "content": "Tell me about SQL joins"},
              {"role": "assistant", "content": "Joins combine rows from two tables."},
              {"role": "user", "content": "Give me an example"}]
    await r.chat('conversation A, ending in "Give me an example"', conv_a, model="meta/llama-3.3-70b-instruct")
    b = await r.chat('conversation B, same last turn, other history', conv_b, model="meta/llama-3.3-70b-instruct")
    r.check(b.headers.get("x-cache") == "MISS", "different history -> no cross-conversation answer")
    jm = await r.chat("the France question again, in JSON mode", q, model="meta/llama-3.3-70b-instruct",
                      response_format={"type": "json_object"})
    r.check(jm.headers.get("x-cache") == "MISS", "JSON mode is never served a cached prose answer")

    # 3 ------------------------------------------------------------- failover
    r.stage(3, "Reliability: retries, instant failover, circuit breaker")
    hiccup = await r.chat("demo/hiccup: first call 503 + Retry-After: 0", "Are you there?", model="demo/hiccup")
    r.check(hiccup.status_code == 200 and hiccup.headers.get("x-gateway-attempts", "").count("=") == 2
            and hiccup.headers.get("x-gateway-provider") == "mock-hiccup",
            "a transient 503 is retried on the same provider")
    revoked = await r.chat("demo/revoked-key: provider answers 401", "Who am I talking to?",
                           model="demo/revoked-key")
    r.check(revoked.status_code == 200 and revoked.headers.get("x-gateway-provider") == "mock-ollama"
            and revoked.headers.get("x-gateway-attempts", "").startswith("mock-revoked:llama-3.3-70b-instruct=401, "),
            "a revoked provider key fails over at once (no retry, no sleep)")
    outage = await r.chat("demo/outage: provider is down (503)", "Is the service up?", model="demo/outage")
    r.check(outage.status_code == 200 and outage.headers.get("x-gateway-provider") == "mock-ollama",
            "a 503 is retried with backoff, then fails over across providers")
    skipped = await r.chat("demo/outage again", "Is the service up now?", model="demo/outage")
    r.check("circuit-open" in skipped.headers.get("x-gateway-attempts", ""),
            "after 2 failures the circuit opens: the dead provider is skipped without a call")
    health = (await r.client.get("/admin/health")).json()
    r.note(f"> GET /admin/health -> status {health['status']}, unhealthy: {', '.join(health['unhealthy_targets'])}")
    r.check(health["targets"]["mock-outage:llama-3.3-70b-instruct"]["state"] == "open",
            "/admin/health reports the open circuit and its last error")

    # 4 ------------------------------------------------------------ streaming
    r.stage(4, "Streaming: SSE pass-through, usage chunk only on request, cached replay")
    payload = {"model": "auto", "stream": True, "messages": [{"role": "user", "content": "Write a haiku about caching."}]}
    s1 = await r.post("/v1/chat/completions", "stream a haiku", payload)
    events = [e for e in _sse(s1) if isinstance(e, dict)]
    text = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e.get("choices"))
    r.note(f"  {len(events)} chunks: {text[:70]}...")
    r.check(s1.status_code == 200 and all(e.get("choices") for e in events),
            "no empty-choices usage chunk unless the client asks for it")
    s2 = await r.post("/v1/chat/completions", "same stream with stream_options.include_usage",
                      {**payload, "stream_options": {"include_usage": True}})
    tail = [e for e in _sse(s2) if isinstance(e, dict)][-1]
    r.check(s2.headers.get("x-cache") == "HIT;EXACT" and tail.get("usage", {}).get("total_tokens", 0) > 0,
            "the repeat is replayed from cache, ending with the usage chunk it asked for")

    # 5 ---------------------------------------------------------- rate limits
    r.stage(5, "Rate limit: key 'burst' allows 3 requests per minute")
    last: Optional[httpx.Response] = None
    for i in range(4):
        last = await r.chat(f"burst request {i + 1}", f"ping {i}", key=BURST_KEY, show=RATE_HEADERS)
    assert last is not None
    r.check(last.status_code == 429 and last.headers.get("x-ratelimit-remaining") == "0"
            and 1 <= int(last.headers.get("retry-after", "0")) <= 20,
            "the 4th request gets 429 with X-RateLimit-* and a computed Retry-After")

    # 6 --------------------------------------------------------------- budget
    r.stage(6, "Budget: key 'tiny-budget' may spend $0.0002 per month")
    essay = "Summarise this report. " + " ".join(["The quarterly numbers were strong."] * 60)
    budget_show = BUDGET_HEADERS
    first_spend = await r.chat("a long prompt to the 70B model", essay, model="meta/llama-3.3-70b-instruct",
                               key=TINY_BUDGET_KEY, show=budget_show)
    second = await r.chat("another request", "And one more thing?", model="meta/llama-3.3-70b-instruct",
                          key=TINY_BUDGET_KEY, show=budget_show)
    r.check(first_spend.status_code == 200 and second.status_code == 402
            and second.headers.get("x-budget-remaining") == "0.0000",
            "once the budget is spent the key gets 402 with X-Budget-* headers")
    forbidden = await r.chat("key 'cheap-only' asks for the 70B model", "hi", model="meta/llama-3.3-70b-instruct",
                             key=CHEAP_ONLY_KEY, show=())
    r.check(forbidden.status_code == 403, "per-key model allow-lists are enforced (403)")

    # 7 ----------------------------------------------------------- embeddings
    r.stage(7, "Embeddings through an alias")
    emb = await r.post("/v1/embeddings", 'model "text-embedding-3-small"',
                       {"model": "text-embedding-3-small", "input": ["semantic caching saves money", "hello"]})
    dims = [len(item["embedding"]) for item in emb.json().get("data", [])]
    r.check(emb.status_code == 200 and dims == [384, 384], f"2 vectors of {dims[0] if dims else '?'} dimensions")

    # 8 ---------------------------------------------------------------- usage
    r.stage(8, "Usage and cost ledger (GET /admin/usage?format=json)")
    summary = (await r.client.get("/admin/usage", params={"format": "json"})).json()
    totals = summary["totals"]
    r.note(f"totals: {totals['requests']} requests, {totals['total_tokens']} tokens, "
           f"{totals['cache_hits']} cache hits, cost ${totals['cost_usd']:.6f}")
    for row in summary["by_key"]:
        r.note(f"  key {row['virtual_key']:<12} requests {row['requests']:>3}  cache hits {row['cache_hits']:>2}  "
               f"cost ${row['cost_usd']:.6f}")
    r.check(totals["cache_hits"] >= 3, "cache hits were recorded at $0")


async def run_demo(
    config_dir: Optional[Path] = None,
    *,
    out: Callable[[str], None] = print,
    verbose: bool = False,
) -> DemoReport:
    from .main import build_app
    from .mock import OfflineGuardTransport

    config_dir = Path(config_dir or DEFAULT_CONFIG_DIR)
    if not (config_dir / "providers.yaml").exists():
        raise FileNotFoundError(
            f"demo config not found in {config_dir} (run from a source checkout, or pass --config-dir)"
        )
    report = DemoReport()
    guard = OfflineGuardTransport()
    quiet = [logging.getLogger(name) for name in ("gateway", "httpx")]
    previous = [logger.level for logger in quiet]
    if not verbose:
        for logger in quiet:
            logger.setLevel(logging.ERROR)
    try:
        with tempfile.TemporaryDirectory(prefix="llm-gateway-demo-") as tmp:
            settings = demo_settings(config_dir, str(Path(tmp) / "demo.db"))
            app = build_app(settings, transport=guard)
            out("llm-gateway offline demo")
            out(f"config: {config_dir}  (mock providers only; any other network access is refused)")
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://gateway.demo") as client:
                    await _stages(_Runner(client, out, report))
    finally:
        for logger, level in zip(quiet, previous):
            logger.setLevel(level)
    report.blocked = list(guard.blocked)
    out("")
    out(f"{report.passed}/{len(report.checks)} checks passed; "
        f"{len(report.blocked)} attempted network calls outside the mock providers.")
    for url in report.blocked:
        out(f"  blocked: {url}")
    return report


def main(config_dir: Optional[str] = None, verbose: bool = False) -> int:
    report = asyncio.run(run_demo(Path(config_dir) if config_dir else None, verbose=verbose))
    return 0 if report.ok else 1
