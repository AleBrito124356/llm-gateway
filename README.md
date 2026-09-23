# llm-gateway

**An OpenAI-compatible LLM gateway — semantic caching, model routing, cross-provider failover with a circuit breaker, per-key rate limits and cost accounting — in front of free NVIDIA NIM, Ollama or any OpenAI-style upstream.**

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-free%20tier-76B900)
![Tests](https://img.shields.io/badge/tests-188%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)

You already have code that talks to the OpenAI API. Point its `base_url` at this
gateway and, without touching anything else, you get: a cache that stops paying
for the same answer twice, automatic failover when an upstream has a bad minute,
one place that enforces per-team rate limits and budgets, and a running tally of
who spent what (and what the cache saved). Hand out **gateway keys** instead of
your real provider keys, and swap Llama-on-NIM for Ollama or OpenAI with a config
edit — the callers never know.

## Try it in one minute — no API key, no network

```bash
git clone https://github.com/AleBrito124356/llm-gateway.git
cd llm-gateway
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

llm-gateway demo
```

The demo boots the real app in-process against [`examples/demo/`](examples/demo),
where every provider is an offline **mock provider** (`type: mock`), and walks
through every feature. Each step is checked, network access is refused (and
counted), and the exit code is non-zero if anything is off. An excerpt:

```text
[2] Two-layer cache: exact hash, then semantic similarity within the same context
    > the identical request again
      200 in 5 ms
        x-cache: HIT;EXACT
      ok   identical request -> HIT;EXACT, $0
    > "Tell me the capital of France, please."
        x-cache: HIT;SEMANTIC
      ok   a reworded question -> HIT;SEMANTIC with the cached answer
    > conversation B, same last turn, other history
        x-cache: MISS
      ok   different history -> no cross-conversation answer

[3] Reliability: retries, instant failover, circuit breaker
    > demo/revoked-key: provider answers 401
        x-gateway-provider: mock-ollama
        x-gateway-attempts: mock-revoked:llama-3.3-70b-instruct=401, mock-ollama:llama3.1=200
      ok   a revoked provider key fails over at once (no retry, no sleep)
    > demo/outage again
        x-gateway-attempts: mock-outage:llama-3.3-70b-instruct=circuit-open, mock-ollama:llama3.1=200
      ok   after 2 failures the circuit opens: the dead provider is skipped without a call
...
23/23 checks passed; 0 attempted network calls outside the mock providers.
```

The same config also runs as a real server, so you can point any OpenAI client
at it — the unmodified [`client_example.py`](client_example.py) works against it:

```bash
llm-gateway serve --config-dir examples/demo            # http://127.0.0.1:8000
python client_example.py                                # in another shell
curl -s "http://127.0.0.1:8000/admin/usage?format=json" # run the example twice: cache hits + saved_usd
```

## Why a gateway

- **Cost control.** A two-layer cache — exact hash, then semantic similarity
  *within the same conversation context* — serves repeated and reworded questions
  for free. Every call is priced and logged, cache savings are priced too, and
  each key has a monthly budget that hard-stops overspend.
- **One endpoint.** Clients only see `/v1/chat/completions`, `/v1/embeddings` and
  `/v1/models`. Which model and provider served the request is the gateway's
  decision, changeable without a client deploy — and without a restart
  (`POST /admin/reload`).
- **Provider portability.** NVIDIA NIM, Ollama, OpenAI and Together are all
  OpenAI-compatible. Route by model name, alias familiar names onto free models,
  or let the virtual `auto` model pick cheap vs strong per request.
- **Reliability.** Transient 5xx, 429 and timeouts are retried with backoff
  (honouring the upstream's `Retry-After`); a revoked key, an unpaid account, a
  retired model or a missing provider key fails over to the next provider at once
  — including a local Ollama model when a hosted API is down — and a circuit
  breaker stops hammering a dead upstream. Streaming gets the same treatment
  before the first byte.
- **Key hygiene.** Upstream keys stay in the gateway's environment; clients
  authenticate with hashed virtual keys you can scope, budget and revoke.

## Architecture

The request pipeline, in order. Each stage can short-circuit the ones after it
(a cache hit skips the upstream entirely; a blown budget never reaches routing).

```mermaid
flowchart LR
    C[OpenAI SDK client] -->|Bearer gateway key| A[Auth virtual key]
    A --> RL[Rate limit and budget guard]
    RL --> R[Router]
    R --> CH{Cache lookup}
    CH -->|exact or semantic hit| HIT[Return cached response, X-Cache HIT]
    CH -->|miss| FB[Failover executor + circuit breaker]
    FB --> P1[NVIDIA NIM primary]
    FB -. retry, fail over, skip open circuits .-> P2[Ollama or OpenAI fallback]
    P1 --> ACC[Accounting: tokens and cost]
    P2 --> ACC
    ACC --> STORE[Write to cache]
    STORE --> OUT[Return response, X-Cache MISS]
```

The virtual `auto` model resolves to a concrete model by cheap heuristics before
the cache is even consulted:

```mermaid
flowchart TD
    Q[Request to model auto] --> S[Classify system plus user messages]
    S --> C1{Contains code}
    C1 -->|yes| STRONG[meta/llama-3.3-70b-instruct]
    C1 -->|no| C2{JSON mode requested}
    C2 -->|yes| STRONG
    C2 -->|no| C3{Prompt longer than threshold}
    C3 -->|yes| STRONG
    C3 -->|no| CHEAP[meta/llama-3.1-8b-instruct]
```

Every routing decision is logged with the signal that triggered it, e.g.
`route auto -> meta/llama-3.3-70b-instruct (strong: code)`.

Every upstream failure is classified before the executor decides what to do:

| Upstream result | Class | What the gateway does |
| --- | --- | --- |
| 5xx, 408, 429, timeout, connection error | RETRY | Retry the same target up to `MAX_RETRIES_PER_TARGET` times with exponential backoff. An upstream `Retry-After` is honoured when it is at most `BACKOFF_CAP_SECONDS`; a longer one fails over immediately. Then fail over. |
| 401, 402, 403, 404, provider key not set | FAILOVER | Next target at once — no retry, no sleep. The caller never sees a 401 for a *provider* key problem. |
| 400, 413, 422 and other 4xx | FATAL | The caller's request is wrong for every target: returned as-is. |
| Target with an open circuit | skipped | After `BREAKER_FAILURE_THRESHOLD` consecutive failures a target is skipped without a call; after `BREAKER_COOLDOWN_SECONDS` one probe request decides whether it is back. |

When every target fails the caller gets `502` (or `503` + `Retry-After` when every
circuit is open), and `X-Gateway-Attempts` lists each hop either way.

## Quickstart with a real (free) NVIDIA NIM key

```bash
pip install -e ".[dev]"
cp .env.example .env
# Get a free NVIDIA NIM key (starts with nvapi-) in ~2 minutes at build.nvidia.com
# and paste it into .env as NVIDIA_API_KEY.

llm-gateway check-config     # validates the four config files and their cross-references
llm-gateway serve            # loads ./.env, listens on http://127.0.0.1:8000
```

Interactive docs are at `/docs`. Without the package installed, the same server
runs with `uvicorn --factory app.main:create_app --env-file .env` (the classic
`uvicorn app.main:app --env-file .env` still works). Use `--host 0.0.0.0` to
listen on all interfaces.

The shipped `keys.yaml` includes one demo virtual key so the examples work out of
the box:

```
sk-gw-demo-000-not-a-real-secret
```

`check-config` warns while it is enabled. Mint your own and remove the demo one
before any real use:

```bash
llm-gateway keys generate --name my-app --rpm 120 --budget 25 --models "meta/*"
# prints the key once, plus the keys.yaml entry to paste in
```

## Usage

### With the OpenAI SDK — change only the base URL

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1",
                api_key="sk-gw-demo-000-not-a-real-secret")

resp = client.chat.completions.create(
    model="auto",  # or a concrete model like meta/llama-3.3-70b-instruct
    messages=[{"role": "user", "content": "Name three uses for an LLM gateway."}],
)
print(resp.model)                    # the model that actually answered
print(resp.choices[0].message.content)
```

Chat, streaming, `auto` routing and embeddings are all shown in
[`client_example.py`](client_example.py).

### With curl

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-gw-demo-000-not-a-real-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"capital of France?"}]}' -i
```

Response headers tell you what happened (real output against `examples/demo`):

```
x-ratelimit-limit: 600
x-ratelimit-remaining: 599
x-ratelimit-reset: 1790181625
x-budget-limit: 5.0000
x-budget-spent: 0.0001
x-budget-remaining: 4.9999
x-budget-reset: 1790812800
x-cache: MISS
x-gateway-model: meta/llama-3.1-8b-instruct
x-gateway-route: default
x-gateway-provider: mock-nim
x-upstream-model: meta/llama-3.1-8b-instruct
x-gateway-attempts: mock-nim:meta/llama-3.1-8b-instruct=200
```

| Header | Meaning |
| --- | --- |
| `X-Cache` | `MISS`, `HIT;EXACT` or `HIT;SEMANTIC` (plus `X-Cache-Similarity` on hits) |
| `X-Gateway-Model` | the gateway model the request resolved to (after aliases / `auto`) |
| `X-Gateway-Route` | why: `alias`, or the `auto` signal (`code`, `json_mode`, `long_prompt`, `default`) |
| `X-Gateway-Provider`, `X-Upstream-Model` | who actually answered (after any failover) |
| `X-Gateway-Attempts` | every hop: `provider:model=status`, or `no-key`, `timeout`, `connect-error`, `circuit-open` |
| `X-RateLimit-Limit/Remaining/Reset` | the key's token bucket (on every response) |
| `X-Budget-Limit/Spent/Remaining/Reset` | the key's monthly budget (keys with a budget only) |
| `Retry-After` | on 429 (from the bucket's refill rate), 402 (until the budget resets) and 503 (until the first open circuit accepts a probe) |

Send the identical request again and `X-Cache` flips to `HIT;EXACT`; a reworded
question with the same conversation context flips it to `HIT;SEMANTIC` once its
embedding is at least `CACHE_SIMILARITY_THRESHOLD`-similar. Hits cost nothing and
return in milliseconds. To bypass the cache for one request, send
`"cache": false`; the gateway consumes that field and never forwards it upstream.

### Streaming

Streams are relayed as-is. The gateway always asks the upstream for a usage chunk
(it needs the token counts for accounting), but forwards that final
empty-`choices` chunk only if the caller asked for it with
`stream_options: {"include_usage": true}` — exactly like OpenAI. The upstream
stream is opened through the same retry/failover/breaker path *before* the
response starts, so a dead chain is a proper HTTP error and the headers name the
provider that is streaming. A cached answer is replayed as a stream when a
streaming request hits the cache.

### Rate limits and budgets

A key over its request-per-minute bucket gets `429` with `X-RateLimit-*` headers
and a `Retry-After` computed from the bucket's refill rate. A key that has spent
its `monthly_budget_usd` gets `402` with `X-Budget-*` headers and a `Retry-After`
until the budget window (UTC month) resets. Both are enforced before any upstream
call. From the demo:

```
> burst request 4                        > another request (key 'tiny-budget')
  429                                      402
    x-ratelimit-limit: 3                     x-budget-limit: 0.0002
    x-ratelimit-remaining: 0                 x-budget-spent: 0.0006
    retry-after: 20                          x-budget-remaining: 0.0000
```

### Usage, cost and savings dashboard

```bash
curl -s "http://localhost:8000/admin/usage?format=json" | python -m json.tool
```

Real output after `llm-gateway demo` (trimmed to one row per list):

```json
{
  "totals": { "requests": 22, "total_tokens": 1666, "cost_usd": 0.0010256,
              "cache_hits": 3, "saved_usd": 0.000102 },
  "by_key": [ { "virtual_key": "demo", "requests": 18, "prompt_tokens": 304, "completion_tokens": 621,
                "total_tokens": 925, "cost_usd": 0.00046454, "cache_hits": 3, "saved_usd": 0.000102 } ],
  "by_model": [ { "model": "meta/llama-3.3-70b-instruct", "provider": "mock-nim", "requests": 11,
                  "total_tokens": 1190, "cost_usd": 0.000972, "cache_hits": 2, "saved_usd": 9.9e-05 } ],
  "cache": { "entries": 18, "semantic_entries": 18, "expired_pending_purge": 0,
             "max_entries": 10000, "ttl_seconds": 86400, "similarity_threshold": 0.85 }
}
```

Open `http://localhost:8000/admin/usage` in a browser for the same data as an
HTML page broken down by key and by model. The ledger bills the model that
actually answered, so a failover to a free local model is recorded at its price.

`saved_usd` is what the cache hits would have cost upstream: each hit is recorded
with its tokens and `cost_usd = 0`, and the dashboard prices those tokens at the
current `pricing.json` rates with `app.cache.estimated_savings`. For `H` hits
averaging `p` prompt and `c` completion tokens, at prices `pi` and `po` per
million:

```
saved_usd = H * (p / 1e6 * pi + c / 1e6 * po)
```

### Admin endpoints

| Endpoint | What it does |
| --- | --- |
| `GET /admin/usage` | the ledger above (HTML, or JSON with `?format=json`) |
| `GET /admin/health` | per target: circuit state, consecutive failures, last status and error, seconds until the next probe; per provider: whether its key is set (never the key) |
| `POST /admin/reload` | re-reads `providers.yaml`, `routing.yaml`, `keys.yaml` and `pricing.json`, validates them like `check-config`, and swaps them in atomically. A config with errors is refused (`400` with the problem list) and the running one is kept. In-flight requests finish on the config they started with; rate-limit buckets, breaker state, cache and ledger survive. |

Set `ADMIN_TOKEN` in production and send it as the `X-Admin-Token` header
(compared in constant time). Without it the admin endpoints are open, and the
gateway logs a warning at startup.

## Configuration

Three YAML files and one JSON file, read from `CONFIG_DIR` (default: the current
directory). Edit them and apply the change with `POST /admin/reload`, no restart
needed. Check them any time with `llm-gateway check-config`: it validates every
cross-reference (aliases, fallbacks, `auto` models and rules, `EMBED_MODEL`), key
hashes, names, rpm and budgets, allow-list globs, pricing coverage and unset
provider key variables, and exits 1 on errors (`--strict` also fails on
warnings, `--json` for CI). The same problems are logged when the gateway starts.

### `providers.yaml` — where requests can go

Each provider is an OpenAI-compatible upstream: a `base_url` and the name of the
environment variable holding its key. Each model maps to a provider and,
optionally, a different `upstream_model` name, per-route cache toggles and
`extra_body` fields merged into every upstream call (the caller's own fields win).

```yaml
providers:
  nvidia:
    base_url: https://integrate.api.nvidia.com/v1
    api_key_env: NVIDIA_API_KEY
  ollama:
    base_url: http://localhost:11434/v1
    api_key_env: OLLAMA_API_KEY
    default_api_key: ollama          # sent when the env var is unset

models:
  meta/llama-3.3-70b-instruct: { provider: nvidia }
  nvidia/nv-embedqa-e5-v5:
    provider: nvidia
    extra_body: { input_type: query, truncate: END }   # NVIDIA's retrieval embedders require input_type
  ollama/llama3.1:
    provider: ollama
    upstream_model: llama3.1
    semantic_cache: false            # do not reuse fuzzy matches for this model
```

A provider with `type: mock` is answered in-process by the gateway itself —
deterministic completions with realistic usage, JSON mode, tool calls, SSE
streaming and embeddings — so you can test clients, routing and failover offline.
Mock providers accept failure injection:

```yaml
providers:
  flaky:
    type: mock
    fail_status: 503      # every request fails with this status ...
    fail_first_n: 2       # ... or only the first N requests, then it recovers
    retry_after: 1        # Retry-After header on injected failures
    latency_ms: 150       # delay every response
    stream_delay_ms: 20   # delay between streamed chunks
```

### `routing.yaml` — how a model name resolves

```yaml
auto:                                # the virtual model
  long_prompt_chars: 2000
  models: { cheap: meta/llama-3.1-8b-instruct, strong: meta/llama-3.3-70b-instruct }
  rules:                             # first match wins
    - { if: code, use: strong }
    - { if: json_mode, use: strong }
    - { if: long_prompt, use: strong }
    - { if: default, use: cheap }

aliases:                             # make existing OpenAI code just work
  gpt-3.5-turbo: meta/llama-3.1-8b-instruct
  gpt-4: meta/llama-3.3-70b-instruct

fallbacks:                           # ordered failover chain, may cross providers
  meta/llama-3.3-70b-instruct:
    - meta/llama-3.1-8b-instruct
    - ollama/llama3.1
```

### `keys.yaml` — who can call, and how much

Only SHA-256 hashes are stored, so the file is safe to commit. Each key has its
own rate limit, optional monthly budget, and a glob-matched allow-list of models
(`GET /v1/models` lists exactly what a key may call). Quote the hash: YAML reads
an unquoted all-digit value as a number (`check-config` catches it).

```yaml
keys:
  - name: demo
    key_hash: "db08fd09bef90acab8dd92fb2e3ee762d65248e412310a2b04a95deb1e970603"
    rpm: 60
    monthly_budget_usd: 5.0
    allowed_models: ["*"]
```

### `pricing.json` — how cost is computed

USD per 1,000,000 tokens, per model. NVIDIA NIM's free tier costs nothing; the
shipped reference prices approximate what the same tokens would cost on a paid
Llama host so the ledger stays meaningful. Set any model to `0` to track tokens
only. Local Ollama models are free. Models without an entry use `default`
(`check-config` lists them).

### Environment variables

Documented in [`.env.example`](.env.example): provider keys, `CONFIG_DIR` and
per-file paths, `GATEWAY_DB`, `EMBED_MODEL` (any model in `providers.yaml`, or the
built-in offline `local/hash`), cache settings (`CACHE_TTL_SECONDS`,
`CACHE_SIMILARITY_THRESHOLD`, `CACHE_MAX_ENTRIES`, `CACHE_PURGE_INTERVAL_SECONDS`),
reliability (`MAX_RETRIES_PER_TARGET`, `BACKOFF_BASE_SECONDS`,
`BACKOFF_CAP_SECONDS`, `REQUEST_TIMEOUT_SECONDS`, `BREAKER_FAILURE_THRESHOLD`,
`BREAKER_COOLDOWN_SECONDS`) and auth (`REQUIRE_AUTH`, `ADMIN_TOKEN`).

## Command line

```text
llm-gateway serve [--host 127.0.0.1] [--port 8000] [--config-dir DIR] [--env-file .env] [--reload] [--strict]
llm-gateway demo [--config-dir examples/demo] [--verbose]
llm-gateway check-config [--config-dir DIR] [--strict] [--json]
llm-gateway keys generate --name NAME [--rpm 60] [--budget USD] [--models GLOB ...]
llm-gateway keys hash sk-gw-...
```

`python -m app ...` runs the same CLI without installing the package, and
`python -m app.keys ...` still works.

## Caveats — when not to cache

Caching trades freshness for cost and latency. Know the trade:

- **Semantic staleness.** The semantic layer returns a *previous* answer for a
  *similar* question. Two prompts can be 0.93 cosine-similar and still want
  different answers ("weather in Paris **today**"). Tune
  `CACHE_SIMILARITY_THRESHOLD` up for safety, down for hit rate, and set the TTL
  to how long an answer stays valid for your domain.
- **What "similar" means here.** Only the final user turn is compared, and only
  against requests whose *context* is byte-identical: same model, system prompt,
  earlier turns and output parameters (`response_format`, `tools`, `max_tokens`,
  `temperature`, `stop`, `seed`, ...). So a JSON-mode request is never served a
  prose answer, and two conversations that both end in "Give me an example" never
  trade answers. Rows written by older versions have no context and are only
  used for exact hits.
- **Time- and user-sensitive prompts.** Anything with "now", "latest", live data
  or per-user context should not be semantically cached. Disable it per model
  with `semantic_cache: false`, or per request by sending `"cache": false`.
- **Tool/function calls** skip the semantic layer automatically — reusing a
  tool-call plan across differently-worded requests is rarely correct.
- **High-temperature sampling** exists to produce variety; a cache hands back the
  same sample every time. Exact-cache such calls only if that is what you want.
- **The embedder matters.** `local/hash` (and the mock providers) use a
  feature-hashing embedder: great for tests, demos and paraphrases that reuse the
  same words, no substitute for a neural embedder on real traffic. If the
  embedder fails, the cache degrades to exact-only instead of turning off.
- **Embedding failover can change dimensions.** If `/v1/embeddings` fails over to
  a different model, the vectors come from that model (named in
  `X-Gateway-Model`) and may not be comparable with your stored ones; leave
  embedding models without fallbacks if that matters.
- **Unverified live details.** The NVIDIA `input_type`/`truncate` fields follow
  NVIDIA's API documentation; this repository's tests run fully offline and never
  call NIM.
- **In-memory state is per process.** Rate-limit buckets and circuit breakers
  live in memory. Behind multiple replicas, move them and the cache to Redis — the
  store interfaces are small on purpose.

## Project structure

```
llm-gateway/
├── app/
│   ├── main.py         # FastAPI app + create_app factory: chat, embeddings, models, /admin/*
│   ├── cli.py          # llm-gateway serve | demo | check-config | keys
│   ├── checks.py       # cross-reference validation of the four config files
│   ├── providers.py    # provider registry, model routes, extra_body, mock providers
│   ├── router.py       # exact/alias/auto routing + failover chain (routing.yaml)
│   ├── cache.py        # two-layer exact + context-aware semantic cache over SQLite
│   ├── limits.py       # token-bucket rate limiting + monthly budget guard
│   ├── accounting.py   # token & cost ledger, usage and savings summary (pricing.json)
│   ├── fallback.py     # retry/failover/fatal classes + circuit breaker
│   ├── upstream.py     # httpx calls to OpenAI-compatible upstreams (incl. SSE)
│   ├── mock.py         # offline mock provider, hash embedder, gateway transport
│   ├── demo.py         # the scripted offline demo
│   ├── keys.py         # hashed virtual-key auth + key-minting CLI
│   ├── schemas.py      # OpenAI-compatible request/response models
│   ├── config.py       # env-driven settings
│   └── db.py           # thin SQLite wrapper with in-place migrations
├── examples/demo/      # all-mock config: the demo, and a keyless server for clients
├── tests/              # 188 tests, all offline
├── client_example.py   # the OpenAI SDK pointed at the gateway
├── providers.yaml  routing.yaml  keys.yaml  pricing.json
├── pyproject.toml  CHANGELOG.md
├── Dockerfile  docker-compose.yml
└── requirements.txt
```

## Run the tests

```bash
pip install -e ".[dev]"            # or: pip install -r requirements.txt
pytest -q                          # 188 tests, no network, no keys
coverage run -m pytest -q && coverage report
```

The suite drives the real FastAPI app over HTTP against a recording fake upstream
and the mock providers: auth, validation, routing, both cache layers and their
isolation rules, streaming, failover classes, the circuit breaker, limits and
budgets with their headers, the ledger and savings, hot reload, the CLI, config
validation, the demo, and the official OpenAI SDK (streaming included) against
the demo config.

## Docker

```bash
docker compose up --build                          # reads provider keys from your .env
docker compose run --rm gateway python -m app demo # the offline demo inside the image
```

The SQLite cache and usage ledger persist in a named volume; the four config
files are mounted read-only so you can edit routing without rebuilding, then
apply it with `POST /admin/reload`. Single-file bind mounts pin the original
inode: if your editor replaces files on save, the container keeps seeing the old
content, so mount a directory and point `CONFIG_DIR` at it, or restart the
container.

## Related projects

- **[ollama-local-llm-kit](https://github.com/AleBrito124356/ollama-local-llm-kit)** — Run LLMs locally with Ollama and flip to free NVIDIA NIM with one flag. A natural upstream and fallback target for this gateway.
- **[nim-free-api-quickstarts](https://github.com/AleBrito124356/nim-free-api-quickstarts)** — Minimal quickstarts for every free NVIDIA NIM capability. Get your `nvapi-` key working before you put the gateway in front of it.
- **[llm-eval-toolkit](https://github.com/AleBrito124356/llm-eval-toolkit)** — Prompt regression testing and evaluation. Use it to prove a cheaper routed model is good enough before you switch traffic.
- **[observability-starter](https://github.com/AleBrito124356/observability-starter)** — OpenTelemetry traces, Prometheus metrics and structured logs on FastAPI. Wire it in to graph latency, cache-hit rate and cost over time.

## License

MIT © 2026 Alejandro Brito. See [LICENSE](LICENSE).
