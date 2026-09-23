"""Static validation of the four config files (``llm-gateway check-config``).

Loading the files only proves they parse. This module also checks that they
agree with each other, so a typo is caught before the first request instead of
surfacing later as a 404, a skipped fallback or a silently default price:

* every alias, fallback, ``auto`` model and the ``EMBED_MODEL`` exists in
  providers.yaml (``local/hash`` is built in);
* ``auto`` rules use known signals and tiers;
* every key has a unique name and a unique, 64-hex-character ``key_hash``
  (quoted: YAML reads an unquoted all-digit hash as a number), a positive
  ``rpm`` and a non-negative budget; allow-list globs match a model; the
  published demo key is flagged so it is not left enabled by accident;
* every routed model has an explicit price;
* the provider API-key environment variables that models depend on are set.

Problems are ``error`` (the config is wrong) or ``warning`` (it works, but
probably not the way you meant). ``POST /admin/reload`` refuses a config with
errors, and the gateway logs every problem at startup.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import yaml

from .config import Settings
from .mock import LOCAL_HASH_MODEL
from .providers import ConfigError, ProviderRegistry
from .router import RoutingConfig

ERROR = "error"
WARNING = "warning"

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
# SHA-256 of the demo key published in the README (sk-gw-demo-000-not-a-real-secret).
PUBLIC_DEMO_KEY_HASH = "db08fd09bef90acab8dd92fb2e3ee762d65248e412310a2b04a95deb1e970603"
_SIGNALS = {"code", "json_mode", "long_prompt", "default"}
_TIERS = {"cheap", "strong"}


@dataclass
class Problem:
    level: str
    file: str
    message: str

    def __str__(self) -> str:
        tag = "ERROR" if self.level == ERROR else "WARN "
        return f"{tag}  {self.file}: {self.message}"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _load_yaml(path: str, name: str, problems: list[Problem], required: bool) -> Optional[dict]:
    if not os.path.exists(path):
        if required:
            problems.append(Problem(ERROR, name, f"file not found: {path}"))
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        problems.append(Problem(ERROR, name, f"cannot be parsed: {exc}"))
        return None
    if not isinstance(data, dict):
        problems.append(Problem(ERROR, name, "top level must be a mapping"))
        return None
    return data


def validate_config(settings: Settings, env: Optional[Mapping[str, str]] = None) -> list[Problem]:
    """Every problem found in the config ``settings`` points at."""
    env = os.environ if env is None else env
    problems: list[Problem] = []
    p_name, r_name = os.path.basename(settings.providers_file), os.path.basename(settings.routing_file)
    k_name, pr_name = os.path.basename(settings.keys_file), os.path.basename(settings.pricing_file)

    # ------------------------------------------------------------ providers
    registry: Optional[ProviderRegistry] = None
    raw_providers = _load_yaml(settings.providers_file, p_name, problems, required=True)
    if raw_providers is not None:
        try:
            registry = ProviderRegistry.from_dict(raw_providers)
        except (ConfigError, TypeError, ValueError) as exc:
            problems.append(Problem(ERROR, p_name, str(exc)))
    models = set(registry.models) if registry else set()

    def known(model: str) -> bool:
        return registry is None or model in models  # no cross-checks without a registry

    # -------------------------------------------------------------- routing
    raw_routing = _load_yaml(settings.routing_file, r_name, problems, required=True)
    routing: Optional[RoutingConfig] = None
    if raw_routing is not None:
        try:
            routing = RoutingConfig.from_dict(raw_routing)
        except (TypeError, ValueError, AttributeError) as exc:
            problems.append(Problem(ERROR, r_name, f"invalid structure: {exc}"))
    if routing is not None:
        # Without an `auto:` section the built-in defaults apply; not using
        # `auto` is fine, so unknown defaults are only a warning.
        auto_level = ERROR if (raw_routing or {}).get("auto") else WARNING
        for tier, model in (("cheap", routing.auto.cheap_model), ("strong", routing.auto.strong_model)):
            if not known(model):
                problems.append(Problem(
                    auto_level, r_name,
                    f"auto.models.{tier} '{model}' is not a model in {p_name}"
                    + ("" if auto_level == ERROR else " (no auto: section, requests to model 'auto' will 404)"),
                ))
        for index, rule in enumerate(routing.auto.rules):
            if not isinstance(rule, dict) or rule.get("if") not in _SIGNALS or rule.get("use") not in _TIERS:
                problems.append(Problem(
                    ERROR, r_name,
                    f"auto.rules[{index}] must be {{if: {'|'.join(sorted(_SIGNALS))}, use: cheap|strong}}, got {rule!r}",
                ))
        for alias, target in routing.aliases.items():
            if not known(target):
                problems.append(Problem(ERROR, r_name, f"alias '{alias}' -> '{target}' is not a model in {p_name}"))
            elif alias in models:
                problems.append(Problem(WARNING, r_name, f"alias '{alias}' shadows the model of the same name"))
        for model, chain in routing.fallbacks.items():
            if not known(model):
                problems.append(Problem(ERROR, r_name, f"fallbacks key '{model}' is not a model in {p_name}"))
            for fallback in chain:
                if not known(fallback):
                    problems.append(Problem(
                        ERROR, r_name, f"fallback '{model}' -> '{fallback}' is not a model in {p_name}",
                    ))

    if settings.semantic_cache_enabled and settings.cache_enabled:
        if settings.embed_model != LOCAL_HASH_MODEL and not known(settings.embed_model):
            problems.append(Problem(
                ERROR, "EMBED_MODEL",
                f"'{settings.embed_model}' is not a model in {p_name} (use one that is, or local/hash)",
            ))

    # ----------------------------------------------------------------- keys
    raw_keys = _load_yaml(settings.keys_file, k_name, problems, required=False)
    if raw_keys is None and not os.path.exists(settings.keys_file) and settings.require_auth:
        problems.append(Problem(WARNING, k_name, "no keys file: every request will get 401 (REQUIRE_AUTH=true)"))
    names: set[str] = set()
    hashes: set[str] = set()
    for index, spec in enumerate((raw_keys or {}).get("keys") or []):
        where = f"keys[{index}]"
        if not isinstance(spec, dict):
            problems.append(Problem(ERROR, k_name, f"{where} must be a mapping"))
            continue
        name = spec.get("name")
        where = f"key '{name}'" if name else where
        if not name:
            problems.append(Problem(ERROR, k_name, f"{where} has no name"))
        elif name in names:
            problems.append(Problem(ERROR, k_name, f"duplicate key name '{name}' (usage and limits would merge)"))
        else:
            names.add(name)
        raw_hash = spec.get("key_hash", "")
        key_hash = str(raw_hash).strip().lower()
        if not isinstance(raw_hash, str):
            problems.append(Problem(
                ERROR, k_name,
                f"{where}: key_hash was read as a {type(raw_hash).__name__} ({raw_hash!r}); quote it in YAML",
            ))
        elif not _HEX64.match(key_hash):
            problems.append(Problem(
                ERROR, k_name, f"{where}: key_hash must be 64 hex characters (a SHA-256), got {key_hash!r}",
            ))
        elif key_hash in hashes:
            problems.append(Problem(ERROR, k_name, f"{where}: key_hash is used by another key"))
        else:
            hashes.add(key_hash)
            if key_hash == PUBLIC_DEMO_KEY_HASH:
                problems.append(Problem(
                    WARNING, k_name, f"{where} uses the public demo key from the README; rotate it before real use",
                ))
        rpm = spec.get("rpm", 60)
        if isinstance(rpm, bool) or not isinstance(rpm, int) or rpm <= 0:
            problems.append(Problem(ERROR, k_name, f"{where}: rpm must be a positive integer, got {rpm!r}"))
        budget = spec.get("monthly_budget_usd")
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget < 0):
            problems.append(Problem(ERROR, k_name, f"{where}: monthly_budget_usd must be >= 0, got {budget!r}"))
        allowed = spec.get("allowed_models", ["*"])
        if not isinstance(allowed, list) or not all(isinstance(p, str) for p in allowed):
            problems.append(Problem(ERROR, k_name, f"{where}: allowed_models must be a list of strings"))
        elif registry is not None:
            for pattern in allowed:
                if not any(fnmatch.fnmatch(model, pattern) for model in models):
                    problems.append(Problem(WARNING, k_name, f"{where}: allowed_models '{pattern}' matches no model"))

    # -------------------------------------------------------------- pricing
    prices: dict[str, Any] = {}
    if not os.path.exists(settings.pricing_file):
        problems.append(Problem(WARNING, pr_name, "no pricing file: every request is recorded at $0"))
    else:
        try:
            with open(settings.pricing_file, "r", encoding="utf-8") as handle:
                prices = json.load(handle)
            if not isinstance(prices, dict):
                raise ValueError("top level must be an object")
        except (OSError, ValueError) as exc:
            problems.append(Problem(ERROR, pr_name, f"cannot be parsed: {exc}"))
            prices = {}
        for model, spec in prices.items():
            if model.startswith("_"):
                continue
            if not isinstance(spec, dict):
                problems.append(Problem(ERROR, pr_name, f"'{model}' must be an object with input_per_1m/output_per_1m"))
                continue
            for field_name in ("input_per_1m", "output_per_1m"):
                value = spec.get(field_name, 0)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    problems.append(Problem(ERROR, pr_name, f"'{model}'.{field_name} must be a number >= 0"))
        if registry is not None and prices:
            default = "default" in prices
            for model in sorted(models):
                if model not in prices:
                    fallback = "the 'default' price" if default else "$0"
                    problems.append(Problem(WARNING, pr_name, f"no price for '{model}': it is billed at {fallback}"))

    # ------------------------------------------------------------ provider keys
    if registry is not None:
        used = {route.provider for route in registry.models.values()}
        for name in sorted(used):
            provider = registry.providers[name]
            if provider.is_mock or provider.default_api_key or not provider.api_key_env:
                continue
            if not env.get(provider.api_key_env):
                problems.append(Problem(
                    WARNING, p_name,
                    f"provider '{name}' reads its key from {provider.api_key_env}, which is not set "
                    "(its models will fail over)",
                ))
    return problems


def summarize(problems: list[Problem]) -> tuple[int, int]:
    errors = sum(1 for p in problems if p.level == ERROR)
    return errors, len(problems) - errors
