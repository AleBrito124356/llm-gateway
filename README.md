# llm-gateway

**An OpenAI-compatible LLM gateway — semantic caching, model routing, cross-provider fallback, per-key rate limits and cost accounting — in front of free NVIDIA NIM, Ollama or any OpenAI-style upstream.**

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-free%20tier-76B900)
![Tests](https://img.shields.io/badge/tests-44%20passing-brightgreen)

You already have code that talks to the OpenAI API. Point its `base_url` at this
gateway and, without touching anything else, you get: a cache that stops paying
for the same answer twice, automatic failover when an upstream has a bad minute,
one place that enforces per-team rate limits and budgets, and a running tally of
who spent what. Hand out **gateway keys** instead of your real provider keys, and
swap Llama-on-NIM for Ollama or OpenAI with a config edit — the callers never
know.

## Why a gateway

- **Cost control.** A two-layer cache — exact hash, then semantic similarity —
  serves repeated and paraphrased questions for free. Every call is priced and
  logged; each key has a monthly budget that hard-stops overspend.
- **One endpoint.** Clients only see `/v1/chat/completions` and `/v1/embeddings`.
  Which model and provider served the request is the gateway's decision,
  changeable without a client deploy.
- **Provider portability.** NVIDIA NIM, Ollama, OpenAI and Together are all
  OpenAI-compatible. Route by model name, alias familiar names onto free models,
  or let the virtual `auto` model pick cheap vs strong per request.
- **Reliability.** Transient 5xx and timeouts are retried with backoff, then
  failed over to the next provider — including a local Ollama model when a hosted
  API is down.
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
    CH -->|miss| FB[Fallback executor]
    FB --> P1[NVIDIA NIM primary]
    FB -. retry then fail over .-> P2[Ollama or OpenAI fallback]
    P1 --> ACC[Accounting: tokens and cost]
    P2 --> ACC
    ACC --> STORE[Write to cache]
    STORE --> OUT[Return response, X-Cache MISS]
```

The virtual `auto` model resolves to a concrete model by cheap heuristics before
the cache is even consulted:

```mermaid
flowchart TD
    Q[Request to model auto] --> S[Classify system plus last user message]
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

## Quickstart

```bash
git clone https://github.com/AleBrito124356/llm-gateway.git
cd llm-gateway
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Get a free NVIDIA NIM key (starts with nvapi-) in ~2 minutes at build.nvidia.com
# and paste it into .env as NVIDIA_API_KEY.

uvicorn app.main:app --reload
```

The server listens on `http://localhost:8000`. Interactive docs at `/docs`.

The shipped `keys.yaml` includes one demo virtual key so the examples work out of
the box:

```
sk-gw-demo-000-not-a-real-secret
```

Mint your own and rotate the demo one before any real use:

```bash
python -m app.keys generate --name my-app --rpm 120 --budget 25
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
print(resp.model)                    # the model the router actually used
print(resp.choices[0].message.content)
```

Streaming, `auto` routing and embeddings are all shown in
[`client_example.py`](client_example.py).

### With curl

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-gw-demo-000-not-a-real-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"capital of France?"}]}' -i
```

Response headers tell you what happened:

```
X-Cache: MISS                     # HIT;EXACT or HIT;SEMANTIC on a repeat
X-Gateway-Model: meta/llama-3.1-8b-instruct
X-Gateway-Provider: nvidia
X-RateLimit-Remaining: 59
X-Budget-Remaining: 5.0000
```

Send the same or a reworded question again and `X-Cache` flips to `HIT;SEMANTIC`
— the response now costs nothing and returns in milliseconds.

### Usage and cost dashboard

```bash
curl -s http://localhost:8000/admin/usage?format=json | jq .totals
```

```json
{ "requests": 128, "total_tokens": 41207, "cost_usd": 0.0371, "cache_hits": 44 }
```

Open `http://localhost:8000/admin/usage` in a browser for the same data as an
HTML table broken down by key and by model. Protect it in production by setting
`ADMIN_TOKEN` and sending it as the `X-Admin-Token` header.

### Rate limits and budgets

A key over its request-per-minute bucket gets `429` with `Retry-After` and
`X-RateLimit-*` headers. A key that has spent its `monthly_budget_usd` gets `402`
with `X-Budget-*` headers. Both are enforced before any upstream call.

## Configuration

Three YAML files and one JSON file, all hot-swappable via Docker volume mounts.

### `providers.yaml` — where requests can go

Each provider is an OpenAI-compatible upstream: a `base_url` and the name of the
environment variable holding its key. Each model maps to a provider and,
optionally, a different `upstream_model` name and per-route cache toggles.

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
  ollama/llama3.1:
    provider: ollama
    upstream_model: llama3.1
    semantic_cache: false            # do not reuse fuzzy matches for this model
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
own rate limit, optional monthly budget, and a glob-matched allow-list of models.

```yaml
keys:
  - name: demo
    key_hash: db08fd09bef90acab8dd92fb2e3ee762d65248e412310a2b04a95deb1e970603
    rpm: 60
    monthly_budget_usd: 5.0
    allowed_models: ["*"]
```

### `pricing.json` — how cost is computed

USD per 1,000,000 tokens, per model. NVIDIA NIM's free tier costs nothing; the
shipped reference prices approximate what the same tokens would cost on a paid
Llama host so the ledger stays meaningful. Set any model to `0` to track tokens
only. Local Ollama models are free.

### The cost-savings math

A cache hit avoids one full generation. For `H` hits whose average request was
`p` prompt and `c` completion tokens, at prices `pi` and `po` per million:

```
saved_usd = H * (p / 1e6 * pi + c / 1e6 * po)
```

The `/admin/usage` page counts hits per key and model; `app.cache.estimated_savings`
turns them into dollars.

## Caveats — when not to cache

Caching trades freshness for cost and latency. Know the trade:

- **Semantic staleness.** The semantic layer returns a *previous* answer for a
  *similar* question. Two prompts can be 0.93 cosine-similar and still want
  different answers ("weather in Paris **today**"). Tune
  `CACHE_SIMILARITY_THRESHOLD` up for safety, down for hit rate, and set the TTL
  to how long an answer stays valid for your domain.
- **Time- and user-sensitive prompts.** Anything with "now", "latest", live data
  or per-user context should not be semantically cached. Disable it per model
  with `semantic_cache: false`, or per request by sending `"cache": false`.
- **Tool/function calls** skip the semantic layer automatically — reusing a
  tool-call plan across differently-worded requests is rarely correct.
- **High-temperature sampling** exists to produce variety; a cache hands back the
  same sample every time. Exact-cache such calls only if that is what you want.
- **In-memory rate-limit state** is per process. Behind multiple replicas, move
  the token buckets and cache to Redis — the store interfaces are small on purpose.

## Project structure

```
llm-gateway/
├── app/
│   ├── main.py         # FastAPI app: /v1/chat/completions, /v1/embeddings, /admin/usage
│   ├── providers.py    # provider registry + model->provider resolution (providers.yaml)
│   ├── router.py       # exact/alias/auto routing + failover chain (routing.yaml)
│   ├── cache.py        # two-layer exact + semantic cache over SQLite
│   ├── limits.py       # token-bucket rate limiting + monthly budget guard
│   ├── accounting.py   # token & cost ledger, usage summary (pricing.json)
│   ├── fallback.py     # retry with backoff + cross-provider failover
│   ├── keys.py         # hashed virtual-key auth + key-minting CLI
│   ├── upstream.py     # httpx calls to OpenAI-compatible upstreams (incl. SSE)
│   ├── schemas.py      # OpenAI-compatible request/response models
│   ├── config.py       # env-driven settings
│   └── db.py           # thin SQLite wrapper
├── tests/              # routing, cache, limits, accounting, fallback (no network)
├── client_example.py   # the OpenAI SDK pointed at the gateway
├── providers.yaml  routing.yaml  keys.yaml  pricing.json
├── Dockerfile  docker-compose.yml
└── requirements.txt
```

## Run the tests

```bash
pytest -q          # 44 tests, no network: routing heuristics, cache hit/miss,
                   # rate-limit buckets, cost math, fallback order
```

## Docker

```bash
docker compose up --build       # reads NVIDIA_API_KEY from your .env
```

The SQLite cache and usage ledger persist in a named volume; the four config
files are mounted read-only so you can edit routing without rebuilding.

## Related projects

- **[ollama-local-llm-kit](https://github.com/AleBrito124356/ollama-local-llm-kit)** — Run LLMs locally with Ollama and flip to free NVIDIA NIM with one flag. A natural upstream and fallback target for this gateway.
- **[nim-free-api-quickstarts](https://github.com/AleBrito124356/nim-free-api-quickstarts)** — Minimal quickstarts for every free NVIDIA NIM capability. Get your `nvapi-` key working before you put the gateway in front of it.
- **[llm-eval-toolkit](https://github.com/AleBrito124356/llm-eval-toolkit)** — Prompt regression testing and evaluation. Use it to prove a cheaper routed model is good enough before you switch traffic.
- **[observability-starter](https://github.com/AleBrito124356/observability-starter)** — OpenTelemetry traces, Prometheus metrics and structured logs on FastAPI. Wire it in to graph latency, cache-hit rate and cost over time.

## License

MIT © 2026 Alejandro Brito. See [LICENSE](LICENSE).
