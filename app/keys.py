"""Virtual API-key authentication.

Callers present a *gateway* key (e.g. ``sk-gw-...``). Only its SHA-256 hash is
stored in ``keys.yaml``, so the file is safe to commit and the upstream provider
keys are never exposed to clients. Each key carries its own rate limit, monthly
budget and allowed-model list.

Run this module directly to mint a key:

    python -m app.keys generate --name mobile-app --rpm 120 --budget 25
    python -m app.keys hash sk-gw-existing-key

(``llm-gateway keys ...`` is the same command once the package is installed.)
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import secrets
from dataclasses import dataclass, field
from typing import Optional

import yaml


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_key() -> str:
    return "sk-gw-" + secrets.token_urlsafe(24)


@dataclass
class VirtualKey:
    name: str
    key_hash: str
    rpm: int = 60
    monthly_budget_usd: Optional[float] = None
    allowed_models: list[str] = field(default_factory=lambda: ["*"])

    def allows_model(self, model: str) -> bool:
        return any(fnmatch.fnmatch(model, pattern) for pattern in self.allowed_models)


class KeyStore:
    def __init__(self, keys: list[VirtualKey]) -> None:
        self._by_hash: dict[str, VirtualKey] = {k.key_hash: k for k in keys}

    @classmethod
    def from_file(cls, path: str) -> "KeyStore":
        if not os.path.exists(path):
            return cls([])
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "KeyStore":
        keys: list[VirtualKey] = []
        for spec in data.get("keys") or []:
            budget = spec.get("monthly_budget_usd")
            keys.append(
                VirtualKey(
                    name=spec["name"],
                    # str(): an unquoted all-digit hash is parsed by YAML as an int.
                    key_hash=str(spec["key_hash"]).strip().lower(),
                    rpm=int(spec.get("rpm", 60)),
                    monthly_budget_usd=None if budget is None else float(budget),
                    allowed_models=list(spec.get("allowed_models", ["*"])),
                )
            )
        return cls(keys)

    def authenticate(self, presented_key: str) -> Optional[VirtualKey]:
        if not presented_key:
            return None
        return self._by_hash.get(hash_key(presented_key))

    def __len__(self) -> int:
        return len(self._by_hash)


def add_key_commands(sub: "argparse._SubParsersAction") -> None:
    """Register ``generate`` and ``hash`` on an argparse sub-command group."""
    gen = sub.add_parser("generate", help="mint a new key and print its keys.yaml entry")
    gen.add_argument("--name", default="new-key")
    gen.add_argument("--rpm", type=int, default=60)
    gen.add_argument("--budget", type=float, default=None, help="monthly budget in USD (omit for unlimited)")
    gen.add_argument("--models", nargs="+", default=["*"], metavar="GLOB",
                     help="allowed model globs (default: all models)")

    h = sub.add_parser("hash", help="print the SHA-256 hash of an existing key")
    h.add_argument("key")


def keys_yaml_entry(name: str, key_hash: str, rpm: int, budget: Optional[float],
                    models: list[str]) -> str:
    # Quoted: YAML would read an (astronomically unlikely) all-digit hash as a number.
    lines = [f"  - name: {name}", f'    key_hash: "{key_hash}"', f"    rpm: {rpm}"]
    if budget is not None:
        lines.append(f"    monthly_budget_usd: {budget}")
    lines.append("    allowed_models: [" + ", ".join(f'"{m}"' for m in models) + "]")
    return "\n".join(lines)


def run_key_command(args: argparse.Namespace) -> int:
    if args.command == "generate":
        if args.rpm <= 0:
            print("error: --rpm must be positive")
            return 2
        key = generate_key()
        print(f"# Give this key to the caller (it is shown only once):\n{key}\n")
        print("# Add this entry under 'keys:' in keys.yaml:")
        print(keys_yaml_entry(args.name, hash_key(key), args.rpm, args.budget, args.models))
    elif args.command == "hash":
        print(hash_key(args.key))
    return 0


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.keys", description="Manage gateway virtual keys.")
    sub = parser.add_subparsers(dest="command", required=True)
    add_key_commands(sub)
    return run_key_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(_cli())
