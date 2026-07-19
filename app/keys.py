"""Virtual API-key authentication.

Callers present a *gateway* key (e.g. ``sk-gw-...``). Only its SHA-256 hash is
stored in ``keys.yaml``, so the file is safe to commit and the upstream provider
keys are never exposed to clients. Each key carries its own rate limit, monthly
budget and allowed-model list.

Run this module directly to mint a key:

    python -m app.keys generate --name mobile-app --rpm 120 --budget 25
    python -m app.keys hash sk-gw-existing-key
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
                    key_hash=spec["key_hash"],
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


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Manage gateway virtual keys.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="mint a new key and print its keys.yaml entry")
    gen.add_argument("--name", default="new-key")
    gen.add_argument("--rpm", type=int, default=60)
    gen.add_argument("--budget", type=float, default=None)

    h = sub.add_parser("hash", help="print the SHA-256 hash of an existing key")
    h.add_argument("key")

    args = parser.parse_args()

    if args.command == "generate":
        key = generate_key()
        print(f"# Give this key to the caller (it is shown only once):\n{key}\n")
        print("# Add this entry under 'keys:' in keys.yaml:")
        print(f"  - name: {args.name}")
        print(f"    key_hash: {hash_key(key)}")
        print(f"    rpm: {args.rpm}")
        if args.budget is not None:
            print(f"    monthly_budget_usd: {args.budget}")
        print('    allowed_models: ["*"]')
    elif args.command == "hash":
        print(hash_key(args.key))


if __name__ == "__main__":
    _cli()
