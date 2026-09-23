"""Retry, failover and circuit breaking across upstream targets.

``execute_with_fallback`` walks an ordered list of targets. Every upstream
failure is classified into one of three outcomes:

``RETRY``    5xx, 408, 429, timeouts and connection errors. The same target is
             retried up to ``max_retries_per_target`` times with exponential
             backoff. An upstream ``Retry-After`` is honoured when it fits under
             ``backoff_cap``; when it does not, waiting is pointless and the
             executor fails over at once.
``FAILOVER`` 401/402/403/404 and a missing provider key. Retrying the same
             target cannot help (revoked key, unpaid account, retired model),
             but another provider may be fine, so the executor moves on
             immediately: no retry, no sleep.
``FATAL``    400/413/422 and other 4xx: the caller's request is wrong and every
             target would reject it, so the error is returned as-is.

A shared ``CircuitBreaker`` remembers per-target health across requests. After
``failure_threshold`` consecutive failures a target is *open* and skipped
without a network call; after ``cooldown_seconds`` one request is let through
as a *half-open* probe, which closes the circuit on success or re-opens it on
failure.

``attempt``, ``sleep`` and the breaker's clock are injected so all of this is
testable without a network or a real clock.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("gateway.fallback")

RETRY = "retry"
FAILOVER = "failover"
FATAL = "fatal"

_FAILOVER_STATUSES = frozenset({401, 402, 403, 404})


def classify_status(status: int) -> str:
    """Map an upstream HTTP status to RETRY, FAILOVER or FATAL."""
    if status >= 500 or status in (408, 429):
        return RETRY
    if status in _FAILOVER_STATUSES:
        return FAILOVER
    return FATAL


class UpstreamError(Exception):
    """An upstream call failed.

    ``kind`` (RETRY/FAILOVER/FATAL) drives the executor. For backward
    compatibility it can be derived from ``retryable``: ``True`` means RETRY,
    ``False`` means FAILOVER for auth/not-found statuses and FATAL otherwise.
    ``retry_after`` carries an upstream ``Retry-After`` in seconds, if any.
    ``reason`` is a short label for failures that have no real upstream status
    (``no-key``, ``timeout``, ``connect-error``), used in attempt summaries.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        retryable: Optional[bool] = None,
        kind: Optional[str] = None,
        retry_after: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(f"{status_code}: {message}")
        if kind is None:
            if retryable is None:
                kind = classify_status(status_code)
            elif retryable:
                kind = RETRY
            else:
                kind = FAILOVER if status_code in _FAILOVER_STATUSES else FATAL
        self.status_code = status_code
        self.message = message
        self.kind = kind
        self.retry_after = retry_after
        self.reason = reason
        # Filled in by the executor on its final error so callers can report hops.
        self.attempts: list[AttemptLog] = []

    @property
    def retryable(self) -> bool:
        return self.kind == RETRY


@dataclass
class AttemptLog:
    target: str
    status_code: int
    message: str
    retry: int
    # "ok", "retry", "failover", "fatal" or "circuit-open" (skipped, no call made)
    outcome: str = "error"
    reason: Optional[str] = None

    def label(self) -> str:
        if self.outcome == "circuit-open":
            return f"{self.target}=circuit-open"
        return f"{self.target}={self.reason or self.status_code}"


def summarize_attempts(attempts: list[AttemptLog]) -> str:
    """Compact hop list for the ``X-Gateway-Attempts`` header."""
    return ", ".join(a.label() for a in attempts)


@dataclass
class FallbackResult:
    value: Any
    target: Any
    attempts: list[AttemptLog] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return summarize_attempts(self.attempts)


def backoff_delay(retry: int, base: float, cap: float) -> float:
    return min(cap, base * (2 ** retry))


# --------------------------------------------------------------- breaker

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


@dataclass
class TargetHealth:
    state: str = CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    probe_started: Optional[float] = None
    successes: int = 0
    failures: int = 0
    last_status: Optional[int] = None
    last_error: Optional[str] = None
    last_error_at: Optional[float] = None
    last_success_at: Optional[float] = None


class CircuitBreaker:
    """Per-target closed -> open -> half-open state machine.

    ``clock`` is monotonic time (cooldowns); ``wall_clock`` only timestamps
    events for the health report. A ``failure_threshold`` of 0 disables it.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self.wall_clock = wall_clock
        self._targets: dict[str, TargetHealth] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.failure_threshold > 0

    def _health(self, name: str) -> TargetHealth:
        health = self._targets.get(name)
        if health is None:
            health = self._targets[name] = TargetHealth()
        return health

    def state(self, name: str) -> str:
        with self._lock:
            return self._health(name).state

    def allow(self, name: str) -> bool:
        """May a request be sent to ``name`` now? Starts a probe when half-open."""
        if not self.enabled:
            return True
        with self._lock:
            health = self._health(name)
            now = self.clock()
            if health.state == CLOSED:
                return True
            if health.state == OPEN:
                if now - health.opened_at >= self.cooldown_seconds:
                    health.state = HALF_OPEN
                    health.probe_started = now
                    logger.info("circuit %s half-open: sending a probe request", name)
                    return True
                return False
            # HALF_OPEN: one probe at a time; a probe that never reported back
            # (e.g. a cancelled stream) is replaced after another cooldown.
            if health.probe_started is None or now - health.probe_started >= self.cooldown_seconds:
                health.probe_started = now
                return True
            return False

    def retry_in(self, name: str) -> float:
        """Seconds until ``name`` will accept a probe (0 if it accepts now)."""
        with self._lock:
            health = self._health(name)
            if health.state != OPEN:
                return 0.0
            return max(0.0, self.cooldown_seconds - (self.clock() - health.opened_at))

    def record_success(self, name: str) -> None:
        with self._lock:
            health = self._health(name)
            if health.state != CLOSED:
                logger.info("circuit %s closed: target recovered", name)
            health.state = CLOSED
            health.consecutive_failures = 0
            health.probe_started = None
            health.successes += 1
            health.last_success_at = self.wall_clock()

    def record_failure(self, name: str, status: int, message: str) -> None:
        with self._lock:
            health = self._health(name)
            health.consecutive_failures += 1
            health.failures += 1
            health.last_status = status
            health.last_error = message[:300]
            health.last_error_at = self.wall_clock()
            if not self.enabled:
                return
            reopen = health.state == HALF_OPEN
            trip = health.state == CLOSED and health.consecutive_failures >= self.failure_threshold
            if reopen or trip:
                health.state = OPEN
                health.opened_at = self.clock()
                health.probe_started = None
                logger.warning(
                    "circuit %s open after %d consecutive failures (last: %s); skipping it for %.0fs",
                    name, health.consecutive_failures, status, self.cooldown_seconds,
                )

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            names = list(self._targets)
        report: dict[str, dict[str, Any]] = {}
        for name in sorted(names):
            retry_in = self.retry_in(name)
            with self._lock:
                h = self._targets[name]
                report[name] = {
                    "state": h.state,
                    "consecutive_failures": h.consecutive_failures,
                    "successes": h.successes,
                    "failures": h.failures,
                    "last_status": h.last_status,
                    "last_error": h.last_error,
                    "last_error_at": h.last_error_at,
                    "last_success_at": h.last_success_at,
                    "retry_in_seconds": round(retry_in, 3),
                }
        return report


# -------------------------------------------------------------- executor


async def execute_with_fallback(
    targets: list[Any],
    attempt: Callable[[Any], Awaitable[Any]],
    *,
    max_retries_per_target: int,
    backoff_base: float,
    backoff_cap: float,
    target_name: Callable[[Any], str] = str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    breaker: Optional[CircuitBreaker] = None,
) -> FallbackResult:
    if not targets:
        raise ValueError("no targets to try")

    attempts: list[AttemptLog] = []
    last_error: Optional[UpstreamError] = None
    skipped: list[str] = []

    for target in targets:
        name = target_name(target)
        if breaker is not None and not breaker.allow(name):
            attempts.append(AttemptLog(name, 0, "circuit open", 0, outcome="circuit-open"))
            skipped.append(name)
            logger.info("skipping %s: circuit open", name)
            continue
        for retry in range(max_retries_per_target + 1):
            try:
                value = await attempt(target)
            except UpstreamError as err:
                last_error = err
                attempts.append(
                    AttemptLog(name, err.status_code, err.message, retry, outcome=err.kind, reason=err.reason)
                )
                if err.kind == FATAL:
                    logger.info("caller error from %s, not retrying: %s", name, err.message)
                    err.attempts = attempts
                    raise
                if breaker is not None:
                    breaker.record_failure(name, err.status_code, err.message)
                if err.kind == FAILOVER:
                    logger.warning("%s cannot serve this request (%s); failing over", name, err.status_code)
                    break
                if retry >= max_retries_per_target:
                    logger.warning("target %s exhausted; failing over", name)
                    break
                if breaker is not None and breaker.state(name) == OPEN:
                    logger.warning("circuit for %s opened mid-request; failing over", name)
                    break
                delay = backoff_delay(retry, backoff_base, backoff_cap)
                if err.retry_after is not None:
                    if err.retry_after > backoff_cap:
                        logger.warning(
                            "%s asks to retry after %.1fs (> cap %.1fs); failing over",
                            name, err.retry_after, backoff_cap,
                        )
                        break
                    delay = max(0.0, err.retry_after)
                logger.warning(
                    "retryable error on %s (attempt %d): %s; retrying in %.2fs",
                    name, retry + 1, err.message, delay,
                )
                if delay > 0:
                    await sleep(delay)
            else:
                if breaker is not None:
                    breaker.record_success(name)
                attempts.append(AttemptLog(name, 200, "ok", retry, outcome="ok"))
                return FallbackResult(value=value, target=target, attempts=attempts)

    summary = summarize_attempts(attempts)
    if last_error is None:
        # Every target was skipped by an open circuit: fail fast.
        wait = min((breaker.retry_in(n) for n in skipped), default=0.0) if breaker else 0.0
        error = UpstreamError(
            503,
            f"no healthy upstream: every target has an open circuit ({', '.join(skipped)})",
            kind=RETRY,
            retry_after=wait,
        )
    else:
        error = UpstreamError(
            502,
            f"all targets failed after {sum(1 for a in attempts if a.outcome != 'circuit-open')} attempts: "
            f"{summary}; last error: {last_error.message}",
            kind=FAILOVER,
        )
    error.attempts = attempts
    raise error
