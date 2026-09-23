"""Token and cost accounting.

Every completed request writes a row into the ``usage`` table. Prices come from
``pricing.json`` (editable, per-model, USD per 1M tokens). NVIDIA NIM's free tier
costs nothing; the reference prices ship so the ledger is still meaningful (they
approximate what the same tokens would cost on a paid Llama host) and so you can
price your own hosted upstreams accurately. Set any model to 0 to track tokens
only.

A cache hit is recorded with its tokens but ``cost_usd = 0``. ``summary`` turns
those cached rows back into dollars (``saved_usd``) with
``cache.estimated_savings`` at the current prices: what the hits would have
cost had they gone upstream.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .cache import estimated_savings
from .db import Database


@dataclass
class Price:
    input_per_1m: float
    output_per_1m: float


class Pricing:
    def __init__(self, prices: dict[str, Price], default: Price) -> None:
        self.prices = prices
        self.default = default

    @classmethod
    def from_file(cls, path: str) -> "Pricing":
        if not os.path.exists(path):
            return cls({}, Price(0.0, 0.0))
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Pricing":
        default_raw = data.get("default", {"input_per_1m": 0.0, "output_per_1m": 0.0})
        default = Price(
            float(default_raw.get("input_per_1m", 0.0)),
            float(default_raw.get("output_per_1m", 0.0)),
        )
        prices: dict[str, Price] = {}
        for model, spec in data.items():
            # Skip the "default" entry and any non-dict metadata (e.g. "_comment").
            if model == "default" or not isinstance(spec, dict):
                continue
            prices[model] = Price(
                float(spec.get("input_per_1m", 0.0)),
                float(spec.get("output_per_1m", 0.0)),
            )
        return cls(prices, default)

    def for_model(self, model: str) -> Price:
        return self.prices.get(model, self.default)

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.for_model(model)
        return round(
            prompt_tokens / 1_000_000 * price.input_per_1m
            + completion_tokens / 1_000_000 * price.output_per_1m,
            8,
        )


def approx_tokens(text: str) -> int:
    """Rough token estimate for upstreams that omit a usage block (~4 chars/token)."""
    return max(1, len(text) // 4)


def _month_start_epoch(now: Optional[float] = None) -> float:
    dt = datetime.fromtimestamp(now or time.time(), tz=timezone.utc)
    return datetime(dt.year, dt.month, 1, tzinfo=timezone.utc).timestamp()


def next_month_start_epoch(now: Optional[float] = None) -> float:
    """When the current monthly budget window ends (first instant of next UTC month)."""
    dt = datetime.fromtimestamp(now or time.time(), tz=timezone.utc)
    year, month = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
    return datetime(year, month, 1, tzinfo=timezone.utc).timestamp()


class Accounting:
    def __init__(self, db: Database, pricing: Pricing) -> None:
        self.db = db
        self.pricing = pricing

    def record(
        self,
        *,
        virtual_key: str,
        model: str,
        provider: str,
        endpoint: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool,
    ) -> float:
        total = prompt_tokens + completion_tokens
        # A cache hit incurs no upstream token cost.
        cost = 0.0 if cached else self.pricing.cost(model, prompt_tokens, completion_tokens)
        self.db.execute(
            "INSERT INTO usage (ts, virtual_key, model, provider, endpoint, "
            "prompt_tokens, completion_tokens, total_tokens, cost_usd, cached) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                virtual_key,
                model,
                provider,
                endpoint,
                prompt_tokens,
                completion_tokens,
                total,
                cost,
                1 if cached else 0,
            ),
        )
        return cost

    def month_cost(self, virtual_key: str, now: Optional[float] = None) -> float:
        row = self.db.query_one(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS c FROM usage "
            "WHERE virtual_key = ? AND ts >= ?",
            (virtual_key, _month_start_epoch(now)),
        )
        return float(row["c"]) if row else 0.0

    def savings(self) -> dict[str, Any]:
        """USD not spent thanks to cache hits, per key, per (model, provider) and in total."""
        rows = self.db.query(
            "SELECT virtual_key, model, provider, COUNT(*) AS hits, "
            "SUM(prompt_tokens) AS p, SUM(completion_tokens) AS c "
            "FROM usage WHERE cached = 1 GROUP BY virtual_key, model, provider"
        )
        by_key: dict[str, float] = defaultdict(float)
        by_model: dict[tuple[str, str], float] = defaultdict(float)
        total = 0.0
        for row in rows:
            hits = int(row["hits"])
            price = self.pricing.for_model(row["model"])
            saved = estimated_savings(
                hits, (row["p"] or 0) / hits, (row["c"] or 0) / hits, price.input_per_1m, price.output_per_1m,
            )
            by_key[row["virtual_key"]] += saved
            by_model[(row["model"], row["provider"])] += saved
            total += saved
        return {"by_key": dict(by_key), "by_model": dict(by_model), "total": total}

    def summary(self) -> dict[str, Any]:
        by_key = self.db.query(
            "SELECT virtual_key, COUNT(*) AS requests, "
            "SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens, "
            "SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, "
            "SUM(cached) AS cache_hits FROM usage GROUP BY virtual_key ORDER BY cost_usd DESC"
        )
        by_model = self.db.query(
            "SELECT model, provider, COUNT(*) AS requests, "
            "SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, "
            "SUM(cached) AS cache_hits FROM usage GROUP BY model, provider ORDER BY cost_usd DESC"
        )
        totals = self.db.query_one(
            "SELECT COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, "
            "SUM(cost_usd) AS cost_usd, SUM(cached) AS cache_hits FROM usage"
        )
        saved = self.savings()
        keys = [_row(r) for r in by_key]
        for row in keys:
            row["saved_usd"] = round(saved["by_key"].get(row["virtual_key"], 0.0), 8)
        models = [_row(r) for r in by_model]
        for row in models:
            row["saved_usd"] = round(saved["by_model"].get((row["model"], row["provider"]), 0.0), 8)
        total_row = _row(totals)
        total_row["saved_usd"] = round(saved["total"], 8)
        return {"totals": total_row, "by_key": keys, "by_model": models}


def _row(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    result: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if value is None:
            value = 0
        elif isinstance(value, float):
            value = round(value, 8)  # 6.2399999999999995e-06 -> 6.24e-06
        result[key] = value
    return result
