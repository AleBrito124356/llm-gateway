"""Retry with backoff and cross-provider failover.

``execute_with_fallback`` walks an ordered list of targets. For each target it
retries transient failures (5xx, 429, timeouts, connection errors) up to
``max_retries_per_target`` with exponential backoff, then moves to the next
target. Non-transient failures (4xx client errors) abort immediately - retrying a
malformed request never helps. Total attempts are therefore capped at
``len(targets) * (max_retries_per_target + 1)``.

``attempt`` and ``sleep`` are injected so this is fully testable without a
network or real clock.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("gateway.fallback")


class UpstreamError(Exception):
    """An upstream call failed. ``retryable`` decides retry/failover eligibility."""

    def __init__(self, status_code: int, message: str, *, retryable: bool) -> None:
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.retryable = retryable


@dataclass
class AttemptLog:
    target: str
    status_code: int
    message: str
    retry: int


@dataclass
class FallbackResult:
    value: Any
    target: Any
    attempts: list[AttemptLog] = field(default_factory=list)


def backoff_delay(retry: int, base: float, cap: float) -> float:
    return min(cap, base * (2 ** retry))


async def execute_with_fallback(
    targets: list[Any],
    attempt: Callable[[Any], Awaitable[Any]],
    *,
    max_retries_per_target: int,
    backoff_base: float,
    backoff_cap: float,
    target_name: Callable[[Any], str] = str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FallbackResult:
    if not targets:
        raise ValueError("no targets to try")

    attempts: list[AttemptLog] = []
    last_error: Optional[UpstreamError] = None

    for target in targets:
        name = target_name(target)
        for retry in range(max_retries_per_target + 1):
            try:
                value = await attempt(target)
                return FallbackResult(value=value, target=target, attempts=attempts)
            except UpstreamError as err:
                last_error = err
                attempts.append(
                    AttemptLog(target=name, status_code=err.status_code, message=err.message, retry=retry)
                )
                if not err.retryable:
                    logger.info("non-retryable error on %s: %s", name, err.message)
                    raise
                if retry < max_retries_per_target:
                    delay = backoff_delay(retry, backoff_base, backoff_cap)
                    logger.warning(
                        "retryable error on %s (attempt %d): %s; retrying in %.2fs",
                        name, retry + 1, err.message, delay,
                    )
                    await sleep(delay)
                else:
                    logger.warning("target %s exhausted; failing over", name)

    assert last_error is not None
    summary = "; ".join(f"{a.target}[{a.status_code}]" for a in attempts)
    raise UpstreamError(
        last_error.status_code,
        f"all targets failed after {len(attempts)} attempts: {summary}",
        retryable=False,
    )
