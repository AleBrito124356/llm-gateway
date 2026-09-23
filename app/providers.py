"""Provider registry.

A *provider* is an OpenAI-compatible upstream (base_url + an API key read from
an environment variable). A *model route* maps a gateway-facing model id to a
provider and, optionally, a different upstream model name plus ``extra_body``
fields merged into every upstream call for that model. Everything is loaded
from ``providers.yaml`` so adding a new upstream is a config edit, not a deploy.

``type: mock`` providers are answered in-process by ``app.mock`` (no network,
no key); they accept failure-injection options, see ``MOCK_OPTIONS``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

import yaml

PROVIDER_TYPES = ("openai", "mock")

# Mock-only options: name -> (type, minimum, maximum).
MOCK_OPTIONS: dict[str, tuple[type, float, float]] = {
    "fail_status": (int, 400, 599),
    "fail_first_n": (int, 0, 1_000_000),
    "retry_after": (float, 0, 86_400),
    "latency_ms": (float, 0, 600_000),
    "stream_delay_ms": (float, 0, 60_000),
    "embedding_dims": (int, 8, 8192),
}


class ConfigError(Exception):
    """Raised for malformed or missing provider configuration."""


class UnknownModelError(Exception):
    """Raised when a requested model has no route."""


@dataclass
class Provider:
    name: str
    base_url: str
    api_key_env: str = ""
    timeout: float = 60.0
    # Some upstreams (Ollama) ignore auth; this literal is sent when the env var
    # is unset so the request still carries a syntactically valid header.
    default_api_key: Optional[str] = None
    # "openai" (any OpenAI-compatible HTTP server) or "mock" (in-process, offline).
    type: str = "openai"
    # Failure-injection and shape options for mock providers.
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mock(self) -> bool:
        return self.type == "mock"

    def api_key(self) -> Optional[str]:
        from_env = os.getenv(self.api_key_env) if self.api_key_env else None
        if self.is_mock:
            return from_env or self.default_api_key or "mock"
        return from_env or self.default_api_key

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def embeddings_url(self) -> str:
        return self.base_url.rstrip("/") + "/embeddings"


@dataclass
class ModelRoute:
    model: str
    provider: str
    upstream_model: str
    cache: bool = True
    semantic_cache: bool = True
    # Provider-specific request fields merged under the caller's body, e.g.
    # ``input_type: query`` for NVIDIA's asymmetric embedding models.
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class Target:
    """A concrete, resolved place to send a request."""

    gateway_model: str
    provider: Provider
    upstream_model: str
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.provider.name}:{self.upstream_model}"

    def payload(self, body: dict[str, Any]) -> dict[str, Any]:
        """The upstream request body: extra_body < caller body < upstream model."""
        return {**self.extra_body, **body, "model": self.upstream_model}


@dataclass
class ProviderRegistry:
    providers: dict[str, Provider] = field(default_factory=dict)
    models: dict[str, ModelRoute] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "ProviderRegistry":
        if not os.path.exists(path):
            raise ConfigError(
                f"providers file not found: {path} (set CONFIG_DIR to the directory "
                "holding providers.yaml, routing.yaml, keys.yaml and pricing.json)"
            )
        with open(path, "r", encoding="utf-8") as handle:
            try:
                data = yaml.safe_load(handle) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderRegistry":
        if not isinstance(data, dict):
            raise ConfigError("providers config must be a mapping")
        providers: dict[str, Provider] = {}
        mock_hosts: dict[str, str] = {}
        for name, spec in (data.get("providers") or {}).items():
            if not isinstance(spec, dict):
                raise ConfigError(f"provider '{name}' must be a mapping")
            ptype = spec.get("type", "openai")
            if ptype not in PROVIDER_TYPES:
                raise ConfigError(
                    f"provider '{name}' has unknown type '{ptype}' (expected one of {', '.join(PROVIDER_TYPES)})"
                )
            base_url = spec.get("base_url")
            if not base_url and ptype == "mock":
                slug = re.sub(r"[^a-z0-9-]+", "-", str(name).lower()).strip("-") or "mock"
                base_url = f"http://{slug}.mock.invalid/v1"
            if not base_url:
                raise ConfigError(f"provider '{name}' is missing base_url")
            options = _mock_options(name, ptype, spec)
            providers[name] = Provider(
                name=name,
                base_url=base_url,
                api_key_env=spec.get("api_key_env", "") or "",
                timeout=float(spec.get("timeout", 60.0)),
                default_api_key=spec.get("default_api_key"),
                type=ptype,
                options=options,
            )
            if ptype == "mock":
                host = urlsplit(base_url).netloc.lower()
                if host in mock_hosts:
                    raise ConfigError(f"mock providers '{mock_hosts[host]}' and '{name}' share the host {host}")
                mock_hosts[host] = name

        models: dict[str, ModelRoute] = {}
        for model_id, spec in (data.get("models") or {}).items():
            spec = spec or {}
            if not isinstance(spec, dict):
                raise ConfigError(f"model '{model_id}' must be a mapping")
            provider_name = spec.get("provider")
            if provider_name not in providers:
                raise ConfigError(
                    f"model '{model_id}' references unknown provider '{provider_name}'"
                )
            extra_body = spec.get("extra_body") or {}
            if not isinstance(extra_body, dict):
                raise ConfigError(f"model '{model_id}': extra_body must be a mapping")
            models[model_id] = ModelRoute(
                model=model_id,
                provider=provider_name,
                upstream_model=spec.get("upstream_model", model_id),
                cache=bool(spec.get("cache", True)),
                semantic_cache=bool(spec.get("semantic_cache", True)),
                extra_body=dict(extra_body),
            )
        return cls(providers=providers, models=models)

    def route(self, gateway_model: str) -> ModelRoute:
        route = self.models.get(gateway_model)
        if route is None:
            raise UnknownModelError(gateway_model)
        return route

    def resolve(self, gateway_model: str) -> Target:
        route = self.route(gateway_model)
        provider = self.providers[route.provider]
        return Target(
            gateway_model=gateway_model,
            provider=provider,
            upstream_model=route.upstream_model,
            extra_body=dict(route.extra_body),
        )

    def list_models(self) -> list[str]:
        return sorted(self.models.keys())


def _mock_options(name: str, ptype: str, spec: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for option, (kind, low, high) in MOCK_OPTIONS.items():
        if option not in spec:
            continue
        if ptype != "mock":
            raise ConfigError(f"provider '{name}': '{option}' only applies to providers with type: mock")
        raw = spec[option]
        try:
            value = kind(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"provider '{name}': {option} must be a number, got {raw!r}") from exc
        if isinstance(raw, bool) or (kind is int and float(raw) != value) or not low <= value <= high:
            raise ConfigError(f"provider '{name}': {option} must be between {low:g} and {high:g}, got {raw!r}")
        options[option] = value
    return options
