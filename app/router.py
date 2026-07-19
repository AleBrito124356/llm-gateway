"""Model routing.

Two jobs, both driven by ``routing.yaml``:

1. Resolve the caller-facing model name to a concrete gateway model. Exact
   matches pass through. Static ``aliases`` map familiar names
   (``gpt-3.5-turbo``) onto whatever you actually serve. The virtual ``auto``
   model is resolved by cheap heuristics into a cheap or a strong model.
2. Build the failover *chain*: the primary target followed by any configured
   fallbacks, resolved through the provider registry (which may cross providers).

Every routing decision is logged so you can see, per request, why a model was
chosen.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from .providers import ProviderRegistry, Target, UnknownModelError
from .schemas import ChatMessage

logger = logging.getLogger("gateway.router")

_CODE_HINTS = re.compile(
    r"```|def\s+\w+\s*\(|class\s+\w+|import\s+\w+|function\s+\w+\s*\(|"
    r"</?[a-z]+>|SELECT\s+.+\s+FROM\s+|#include\s|=>|\bconsole\.log\b|\bpublic\s+static\b",
    re.IGNORECASE,
)


@dataclass
class AutoConfig:
    long_prompt_chars: int = 2000
    cheap_model: str = "meta/llama-3.1-8b-instruct"
    strong_model: str = "meta/llama-3.3-70b-instruct"
    # Ordered rules. Each is ``{"if": <signal>, "use": "cheap"|"strong"}``.
    rules: list[dict[str, str]] = field(default_factory=list)


@dataclass
class RoutingConfig:
    auto: AutoConfig
    aliases: dict[str, str]
    fallbacks: dict[str, list[str]]

    @classmethod
    def from_file(cls, path: str) -> "RoutingConfig":
        if not os.path.exists(path):
            raise FileNotFoundError(f"routing file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingConfig":
        auto_raw = data.get("auto") or {}
        models = auto_raw.get("models") or {}
        auto = AutoConfig(
            long_prompt_chars=int(auto_raw.get("long_prompt_chars", 2000)),
            cheap_model=models.get("cheap", "meta/llama-3.1-8b-instruct"),
            strong_model=models.get("strong", "meta/llama-3.3-70b-instruct"),
            rules=list(
                auto_raw.get("rules")
                or [
                    {"if": "code", "use": "strong"},
                    {"if": "json_mode", "use": "strong"},
                    {"if": "long_prompt", "use": "strong"},
                    {"if": "default", "use": "cheap"},
                ]
            ),
        )
        return cls(
            auto=auto,
            aliases=dict(data.get("aliases") or {}),
            fallbacks={k: list(v) for k, v in (data.get("fallbacks") or {}).items()},
        )


def detect_code(text: str) -> bool:
    return bool(_CODE_HINTS.search(text))


def classify(messages: list[ChatMessage], wants_json: bool, cfg: AutoConfig) -> dict[str, Any]:
    """Compute the signals the ``auto`` rules match against."""
    joined = "\n".join(m.text() for m in messages if m.role in {"system", "user"})
    return {
        "chars": len(joined),
        "code": detect_code(joined),
        "json_mode": wants_json,
        "long_prompt": len(joined) > cfg.long_prompt_chars,
    }


class Router:
    def __init__(self, config: RoutingConfig, registry: ProviderRegistry) -> None:
        self.config = config
        self.registry = registry

    def resolve_model(
        self,
        model: str,
        messages: list[ChatMessage],
        wants_json: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Return the concrete gateway model and a decision record for logging."""
        if model == "auto":
            signals = classify(messages, wants_json, self.config.auto)
            tier = "cheap"
            reason = "default"
            for rule in self.config.auto.rules:
                cond = rule.get("if")
                if cond == "default" or signals.get(cond):
                    tier = rule.get("use", "cheap")
                    reason = cond
                    break
            chosen = (
                self.config.auto.strong_model
                if tier == "strong"
                else self.config.auto.cheap_model
            )
            decision = {
                "requested": "auto",
                "resolved": chosen,
                "tier": tier,
                "reason": reason,
                "signals": signals,
            }
            logger.info("route auto -> %s (%s: %s)", chosen, tier, reason)
            return chosen, decision

        if model in self.config.aliases:
            resolved = self.config.aliases[model]
            decision = {"requested": model, "resolved": resolved, "reason": "alias"}
            logger.info("route alias %s -> %s", model, resolved)
            return resolved, decision

        decision = {"requested": model, "resolved": model, "reason": "exact"}
        return model, decision

    def resolve_chain(self, gateway_model: str) -> list[Target]:
        """Primary target plus configured fallbacks, deduplicated, resolved."""
        order = [gateway_model] + self.config.fallbacks.get(gateway_model, [])
        chain: list[Target] = []
        seen: set[str] = set()
        for name in order:
            if name in seen:
                continue
            seen.add(name)
            try:
                chain.append(self.registry.resolve(name))
            except UnknownModelError:
                logger.warning("fallback model '%s' has no route; skipping", name)
        if not chain:
            # Force the original error to surface with a clear message.
            self.registry.resolve(gateway_model)
        return chain
