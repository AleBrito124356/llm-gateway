"""Runtime configuration.

Settings come from environment variables (see .env.example). Paths to the four
config files can be overridden so the same image can serve different deployments.
Every field has a default so tests (and code embedding the gateway) can build a
``Settings`` directly and override only what they care about.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass
class Settings:
    """Process-wide settings, all overridable via environment variables."""

    config_dir: str = "."
    providers_file: str = "providers.yaml"
    routing_file: str = "routing.yaml"
    keys_file: str = "keys.yaml"
    pricing_file: str = "pricing.json"
    db_path: str = "gateway.db"

    cache_enabled: bool = True
    semantic_cache_enabled: bool = True
    cache_ttl_seconds: int = 60 * 60 * 24
    cache_similarity_threshold: float = 0.92
    # Hard cap on stored cache rows (oldest evicted first). 0 disables the cap.
    cache_max_entries: int = 10_000
    # How often expired rows are purged and the cap is enforced.
    cache_purge_interval_seconds: float = 300.0

    embed_model: str = "nvidia/nv-embedqa-e5-v5"
    default_chat_model: str = "meta/llama-3.3-70b-instruct"

    max_retries_per_target: int = 2
    backoff_base_seconds: float = 0.5
    backoff_cap_seconds: float = 8.0
    request_timeout_seconds: float = 60.0

    require_auth: bool = True
    # Protects /admin/*. Unset means the admin endpoints are open (dev only).
    admin_token: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        config_dir: Optional[str] = None,
        **overrides: Any,
    ) -> "Settings":
        """Build settings from the environment.

        ``config_dir`` (or ``CONFIG_DIR``) is where the four config files and the
        default ``gateway.db`` live; each file can still be pointed elsewhere with
        its own variable. ``overrides`` replace individual fields afterwards.
        """
        env = os.environ if env is None else env
        base = config_dir or env.get("CONFIG_DIR") or "."

        def _path(env_name: str, filename: str) -> str:
            return env.get(env_name) or os.path.join(base, filename)

        settings = cls(
            config_dir=base,
            providers_file=_path("PROVIDERS_FILE", "providers.yaml"),
            routing_file=_path("ROUTING_FILE", "routing.yaml"),
            keys_file=_path("KEYS_FILE", "keys.yaml"),
            pricing_file=_path("PRICING_FILE", "pricing.json"),
            db_path=env.get("GATEWAY_DB") or os.path.join(base, "gateway.db"),
            cache_enabled=_env_bool(env, "CACHE_ENABLED", True),
            semantic_cache_enabled=_env_bool(env, "SEMANTIC_CACHE_ENABLED", True),
            cache_ttl_seconds=_env_int(env, "CACHE_TTL_SECONDS", 60 * 60 * 24),
            cache_similarity_threshold=_env_float(env, "CACHE_SIMILARITY_THRESHOLD", 0.92),
            cache_max_entries=_env_int(env, "CACHE_MAX_ENTRIES", 10_000),
            cache_purge_interval_seconds=_env_float(env, "CACHE_PURGE_INTERVAL_SECONDS", 300.0),
            embed_model=env.get("EMBED_MODEL") or "nvidia/nv-embedqa-e5-v5",
            default_chat_model=env.get("NIM_MODEL") or "meta/llama-3.3-70b-instruct",
            max_retries_per_target=_env_int(env, "MAX_RETRIES_PER_TARGET", 2),
            backoff_base_seconds=_env_float(env, "BACKOFF_BASE_SECONDS", 0.5),
            backoff_cap_seconds=_env_float(env, "BACKOFF_CAP_SECONDS", 8.0),
            request_timeout_seconds=_env_float(env, "REQUEST_TIMEOUT_SECONDS", 60.0),
            require_auth=_env_bool(env, "REQUIRE_AUTH", True),
            admin_token=env.get("ADMIN_TOKEN") or None,
        )
        return replace(settings, **overrides) if overrides else settings
