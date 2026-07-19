"""Runtime configuration.

Settings come from environment variables (see .env.example). Paths to the four
config files can be overridden so the same image can serve different deployments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass
class Settings:
    """Process-wide settings, all overridable via environment variables."""

    providers_file: str
    routing_file: str
    keys_file: str
    pricing_file: str
    db_path: str

    cache_enabled: bool
    semantic_cache_enabled: bool
    cache_ttl_seconds: int
    cache_similarity_threshold: float

    embed_model: str
    default_chat_model: str

    max_retries_per_target: int
    backoff_base_seconds: float
    backoff_cap_seconds: float
    request_timeout_seconds: float

    require_auth: bool

    @classmethod
    def from_env(cls) -> "Settings":
        base = os.getenv("CONFIG_DIR", ".")

        def _path(env_name: str, filename: str) -> str:
            return os.getenv(env_name, os.path.join(base, filename))

        return cls(
            providers_file=_path("PROVIDERS_FILE", "providers.yaml"),
            routing_file=_path("ROUTING_FILE", "routing.yaml"),
            keys_file=_path("KEYS_FILE", "keys.yaml"),
            pricing_file=_path("PRICING_FILE", "pricing.json"),
            db_path=os.getenv("GATEWAY_DB", os.path.join(base, "gateway.db")),
            cache_enabled=_env_bool("CACHE_ENABLED", True),
            semantic_cache_enabled=_env_bool("SEMANTIC_CACHE_ENABLED", True),
            cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 60 * 60 * 24),
            cache_similarity_threshold=_env_float("CACHE_SIMILARITY_THRESHOLD", 0.92),
            embed_model=os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"),
            default_chat_model=os.getenv("NIM_MODEL", "meta/llama-3.3-70b-instruct"),
            max_retries_per_target=_env_int("MAX_RETRIES_PER_TARGET", 2),
            backoff_base_seconds=_env_float("BACKOFF_BASE_SECONDS", 0.5),
            backoff_cap_seconds=_env_float("BACKOFF_CAP_SECONDS", 8.0),
            request_timeout_seconds=_env_float("REQUEST_TIMEOUT_SECONDS", 60.0),
            require_auth=_env_bool("REQUIRE_AUTH", True),
        )
