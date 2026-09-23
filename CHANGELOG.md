# Changelog

## 0.2.0

### Fixed

- **Semantic cache could serve the wrong answer.** It embedded only the system
  prompt plus the last user turn and matched on the model alone, so two
  conversations ending in "Give me an example" traded answers, and a JSON-mode
  request could be served a cached prose answer. Semantic hits now require an
  identical `context_key` (model, system prompt, earlier turns and every
  output-shaping parameter), and only the final user turn is embedded.
- **A failing embedder disabled the whole cache.** `store()` awaited the
  embedding before writing, so an embed error also dropped the exact entry.
  The exact entry is now always written, and a miss costs one embedding call
  instead of two (the lookup's vector is reused).
- **The gateway-only `cache` field was forwarded upstream**, where OpenAI
  rejects unknown arguments. It is now consumed by the gateway.
- **Streaming broke `client_example.py`** (`IndexError`): the upstream usage
  chunk (empty `choices`) was forwarded to clients that never asked for it. It
  is now forwarded only with `stream_options.include_usage`.
- **429 and 402 lost their headers.** They now carry `X-RateLimit-*` /
  `X-Budget-*` and a `Retry-After` computed from the token bucket or the budget
  reset, instead of a hardcoded `Retry-After: 1`.
- **A revoked/expired provider key or a retired model (401/403/404) aborted the
  request** with no failover, and was returned as a 401 that the OpenAI SDK
  reports as the caller's own key being invalid. These now fail over at once
  and surface as 502 if nothing else answers.
- **A missing provider key was retried with sleeps** (11.5 s to fail with no
  NIM key). It now fails over immediately; the shipped config reaches a healthy
  Ollama fallback in about 10 ms.
- **The cache table grew without bound.** Expired rows are purged and the table
  is capped at `CACHE_MAX_ENTRIES` periodically.
- **Importing `app.main` read `./providers.yaml` and created `gateway.db`**, so
  it failed outside the repo root. The module-level `app` is now built lazily.
- **NVIDIA's asymmetric embedders require `input_type`.** Models accept an
  `extra_body` in providers.yaml; the shipped config sends
  `input_type: query` (and `truncate: END`) for `nvidia/nv-embedqa-e5-v5`.
- **The shipped `cheap-only` key hash was unquoted**, so YAML read it as the
  integer 0. It is quoted now, and `KeyStore` coerces hashes to strings.
- The app never read `.env` although the Quickstart said to put the key there.
  `llm-gateway serve` now loads it (plain uvicorn: `--env-file .env`).

### Added

- Offline **mock providers** (`type: mock`) with failure injection
  (`fail_status`, `fail_first_n`, `retry_after`, `latency_ms`,
  `stream_delay_ms`), served in-process with no sockets.
- `local/hash`: a built-in offline embedder for the semantic cache.
- `llm-gateway demo` / `python -m app demo`: a scripted, self-checking tour
  of every feature against `examples/demo/`, with network access refused.
- Failure classes (RETRY / FAILOVER / FATAL), upstream `Retry-After` support,
  a per-target **circuit breaker** (`BREAKER_FAILURE_THRESHOLD`,
  `BREAKER_COOLDOWN_SECONDS`), retries and failover for streaming before the
  first byte, and the `X-Gateway-Attempts` header.
- `GET /admin/health` (breaker state and last error per target) and
  `POST /admin/reload` (validated, atomic hot reload of the four config files).
- `saved_usd` per key, per model and in total on `/admin/usage` (JSON and
  HTML), plus cache statistics.
- `pyproject.toml` and the `llm-gateway` command: `serve`, `demo`,
  `check-config`, `keys generate|hash`.
- `llm-gateway check-config`: cross-reference validation of all config files.
- 188 offline tests (was 44); coverage 97% (was 47%).

### Changed (behaviour)

- `GET /v1/models` requires a gateway key (like OpenAI) and lists only the
  models, aliases and `auto` that key may call.
- When every target fails before a stream starts, the caller gets an HTTP
  error (502/503) instead of an SSE error event inside a 200.
- The ledger records the model that actually answered (after a failover),
  not the one requested; cache hits are still recorded against the requested one.
- `X-Budget-*` headers are sent only for keys that have a budget.
- The HTML dashboard escapes names, supports dark mode and scrolls wide tables
  on small screens.
- `Settings` fields all have defaults and `Settings.from_env()` accepts an
  explicit environment mapping, a `config_dir` and overrides.
- `UpstreamError` carries a `kind` (and optional `retry_after` / `reason`);
  `retryable` is derived from it and still accepted by the constructor.
- `docker-compose.yml` no longer requires `NVIDIA_API_KEY` (models without a
  key fail over) and passes `ADMIN_TOKEN` through.
- Cache rows written by 0.1.0 have no context key: they still serve exact hits
  but are never used for semantic hits.

### Removed

- `NIM_MODEL` / `Settings.default_chat_model`: it was never read.
