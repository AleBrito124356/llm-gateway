"""Provider registry.

A *provider* is an OpenAI-compatible upstream (base_url + an API key read from
an environment variable). A *model route* maps a gateway-facing model id to a
provider and, optionally, a different upstream model name plus ``extra_body``
fields merged into every upstream call for that model. Everything is loaded
from ``providers.yaml`` so adding a new upstream is a config edit, not a deploy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


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

    def api_key(self) -> Optional[str]:
        from_env = os.getenv(self.api_key_env) if self.api_key_env else None
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
        for name, spec in (data.get("providers") or {}).items():
            if not isinstance(spec, dict):
                raise ConfigError(f"provider '{name}' must be a mapping")
            if "base_url" not in spec:
                raise ConfigError(f"provider '{name}' is missing base_url")
            providers[name] = Provider(
                name=name,
                base_url=spec["base_url"],
                api_key_env=spec.get("api_key_env", "") or "",
                timeout=float(spec.get("timeout", 60.0)),
                default_api_key=spec.get("default_api_key"),
            )

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
