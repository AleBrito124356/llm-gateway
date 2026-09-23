"""Per-virtual-key rate limiting and monthly budget guard.

Rate limiting is a classic token bucket: each key gets ``rpm`` requests per
minute with burst up to ``rpm``, refilling continuously. Every response carries
``X-RateLimit-Limit/Remaining/Reset``; a refused request gets 429 with the same
headers plus a ``Retry-After`` computed from the bucket's refill rate.

The budget guard is separate: if a key's month-to-date spend (from the
accounting ledger) has reached its ``monthly_budget_usd``, requests are refused
with 402. Keys with a budget get ``X-Budget-Limit/Spent/Remaining/Reset`` on
every response (and ``Retry-After`` on the 402). Buckets are in-memory and
per-process; for multiple replicas move this state to Redis.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenBucket:
    capacity: float
    refill_per_sec: float
    tokens: float
    updated: float

    def _refill(self, now: float) -> None:
        elapsed = now - self.updated
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            self.updated = now

    def try_consume(self, amount: float = 1.0, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.monotonic()
        self._refill(now)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def seconds_until(self, amount: float = 1.0, now: Optional[float] = None) -> float:
        now = now if now is not None else time.monotonic()
        self._refill(now)
        if self.tokens >= amount:
            return 0.0
        if self.refill_per_sec <= 0:
            return float("inf")
        return (amount - self.tokens) / self.refill_per_sec


@dataclass
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int
    # Exact seconds until the next request would be allowed (0 when allowed).
    retry_after: float = 0.0

    def headers(self) -> dict[str, str]:
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(time.time()) + self.reset_seconds),
        }
        if not self.allowed:
            headers["Retry-After"] = str(retry_after_header(self.retry_after))
        return headers


def retry_after_header(seconds: float) -> int:
    """Whole seconds for a Retry-After header: rounded up, at least 1."""
    if math.isinf(seconds) or math.isnan(seconds):
        return 3600
    return max(1, math.ceil(seconds))


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, key: str, rpm: int) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or bucket.capacity != rpm:
                bucket = TokenBucket(
                    capacity=float(rpm),
                    refill_per_sec=rpm / 60.0,
                    tokens=float(rpm),
                    updated=time.monotonic(),
                )
                self._buckets[key] = bucket
            return bucket

    def check(self, key: str, rpm: int, now: Optional[float] = None) -> RateDecision:
        bucket = self._bucket(key, rpm)
        with self._lock:
            allowed = bucket.try_consume(1.0, now=now)
            remaining = int(bucket.tokens)
            wait = 0.0 if allowed else bucket.seconds_until(1.0, now=now)
        reset = retry_after_header(wait) if not allowed else 0
        return RateDecision(
            allowed=allowed, limit=rpm, remaining=remaining, reset_seconds=reset, retry_after=wait,
        )


@dataclass
class BudgetDecision:
    allowed: bool
    limit_usd: float
    spent_usd: float
    # Epoch seconds when the monthly budget window resets (UTC month start).
    resets_at: Optional[float] = None

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    @property
    def unlimited(self) -> bool:
        return math.isinf(self.limit_usd)

    def headers(self, now: Optional[float] = None) -> dict[str, str]:
        if self.unlimited:
            return {}
        headers = {
            "X-Budget-Limit": f"{self.limit_usd:.4f}",
            "X-Budget-Spent": f"{self.spent_usd:.4f}",
            "X-Budget-Remaining": f"{self.remaining_usd:.4f}",
        }
        if self.resets_at is not None:
            headers["X-Budget-Reset"] = str(int(self.resets_at))
            if not self.allowed:
                now = time.time() if now is None else now
                headers["Retry-After"] = str(retry_after_header(self.resets_at - now))
        return headers


def check_budget(
    spent_usd: float,
    monthly_budget_usd: Optional[float],
    resets_at: Optional[float] = None,
) -> BudgetDecision:
    if monthly_budget_usd is None:
        return BudgetDecision(allowed=True, limit_usd=float("inf"), spent_usd=spent_usd)
    return BudgetDecision(
        allowed=spent_usd < monthly_budget_usd,
        limit_usd=monthly_budget_usd,
        spent_usd=spent_usd,
        resets_at=resets_at,
    )
