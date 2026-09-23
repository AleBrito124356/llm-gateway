"""Error classes, Retry-After handling and the circuit breaker (no network, fake clock)."""

from __future__ import annotations

import pytest

from app.fallback import (
    FAILOVER,
    FATAL,
    RETRY,
    CircuitBreaker,
    UpstreamError,
    classify_status,
    execute_with_fallback,
    summarize_attempts,
)


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_status_classification():
    assert [classify_status(s) for s in (500, 502, 503, 504, 408, 429)] == [RETRY] * 6
    assert [classify_status(s) for s in (401, 402, 403, 404)] == [FAILOVER] * 4
    assert [classify_status(s) for s in (400, 413, 422, 409)] == [FATAL] * 4


def test_legacy_retryable_flag_maps_to_kinds():
    assert UpstreamError(503, "x", retryable=True).kind == RETRY
    assert UpstreamError(401, "x", retryable=False).kind == FAILOVER
    assert UpstreamError(400, "x", retryable=False).kind == FATAL
    assert UpstreamError(503, "x", retryable=False).kind == FATAL
    assert UpstreamError(429, "x").retryable


async def test_failover_class_moves_on_without_retry_or_sleep():
    seen, sleeps = [], Sleeps()

    async def attempt(target):
        seen.append(target)
        if target == "a":
            raise UpstreamError(401, "revoked key", kind=FAILOVER)
        return "ok-b"

    result = await execute_with_fallback(
        ["a", "b"], attempt, max_retries_per_target=3, backoff_base=1, backoff_cap=8, sleep=sleeps,
    )
    assert result.value == "ok-b"
    assert seen == ["a", "b"]
    assert sleeps.calls == []
    assert result.summary == "a=401, b=200"


async def test_retry_after_is_honoured_when_under_the_cap():
    sleeps, calls = Sleeps(), {"n": 0}

    async def attempt(target):
        calls["n"] += 1
        if calls["n"] == 1:
            raise UpstreamError(429, "slow down", kind=RETRY, retry_after=1.5)
        return "ok"

    await execute_with_fallback(["a"], attempt, max_retries_per_target=2, backoff_base=0.1,
                                backoff_cap=8, sleep=sleeps)
    assert sleeps.calls == [1.5]


async def test_retry_after_over_the_cap_fails_over_immediately():
    sleeps, seen = Sleeps(), []

    async def attempt(target):
        seen.append(target)
        if target == "a":
            raise UpstreamError(429, "come back in a minute", kind=RETRY, retry_after=60)
        return "ok"

    result = await execute_with_fallback(["a", "b"], attempt, max_retries_per_target=3,
                                         backoff_base=0.5, backoff_cap=8, sleep=sleeps)
    assert result.value == "ok" and seen == ["a", "b"] and sleeps.calls == []


async def test_backoff_sleeps_between_retries_then_reports_every_hop():
    sleeps = Sleeps()

    async def attempt(target):
        raise UpstreamError(503, "down", kind=RETRY)

    with pytest.raises(UpstreamError) as exc:
        await execute_with_fallback(["a"], attempt, max_retries_per_target=2, backoff_base=0.5,
                                    backoff_cap=8, sleep=sleeps)
    assert sleeps.calls == [0.5, 1.0]
    assert exc.value.status_code == 502
    assert exc.value.kind == FAILOVER
    assert summarize_attempts(exc.value.attempts) == "a=503, a=503, a=503"


async def test_fatal_error_keeps_status_and_reports_attempts():
    async def attempt(target):
        raise UpstreamError(422, "bad schema", kind=FATAL)

    with pytest.raises(UpstreamError) as exc:
        await execute_with_fallback(["a", "b"], attempt, max_retries_per_target=2,
                                    backoff_base=0, backoff_cap=0, sleep=Sleeps())
    assert exc.value.status_code == 422
    assert [a.target for a in exc.value.attempts] == ["a"]


def test_breaker_state_machine():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=30, clock=clock, wall_clock=clock)
    assert breaker.allow("t") and breaker.state("t") == "closed"
    breaker.record_failure("t", 503, "down")
    assert breaker.state("t") == "closed"
    breaker.record_failure("t", 503, "down again")
    assert breaker.state("t") == "open"
    assert not breaker.allow("t")
    assert breaker.retry_in("t") == 30
    clock.now += 29
    assert not breaker.allow("t")
    clock.now += 1
    assert breaker.allow("t")  # the half-open probe
    assert breaker.state("t") == "half_open"
    assert not breaker.allow("t")  # only one probe in flight
    breaker.record_failure("t", 500, "still down")
    assert breaker.state("t") == "open"  # a failed probe re-opens at once
    clock.now += 30
    assert breaker.allow("t")
    breaker.record_success("t")
    assert breaker.state("t") == "closed" and breaker.allow("t")
    snap = breaker.snapshot()["t"]
    assert snap["failures"] == 3 and snap["successes"] == 1 and snap["last_status"] == 500
    assert snap["last_error"] == "still down"


def test_breaker_success_resets_the_failure_streak():
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=30)
    breaker.record_failure("t", 503, "x")
    breaker.record_success("t")
    breaker.record_failure("t", 503, "x")
    assert breaker.state("t") == "closed"


def test_breaker_replaces_a_probe_that_never_reported_back():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=10, clock=clock)
    breaker.record_failure("t", 503, "x")
    clock.now += 10
    assert breaker.allow("t")      # probe #1 starts, then vanishes (cancelled stream)
    assert not breaker.allow("t")
    clock.now += 10
    assert breaker.allow("t")      # a new probe is allowed after another cooldown


def test_breaker_threshold_zero_disables_it():
    breaker = CircuitBreaker(failure_threshold=0, cooldown_seconds=10)
    for _ in range(50):
        breaker.record_failure("t", 503, "x")
    assert breaker.allow("t") and breaker.state("t") == "closed"


async def test_open_circuit_is_skipped_without_a_call():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
    breaker.record_failure("a", 503, "down")
    seen = []

    async def attempt(target):
        seen.append(target)
        return f"ok-{target}"

    result = await execute_with_fallback(["a", "b"], attempt, max_retries_per_target=2, backoff_base=0,
                                         backoff_cap=0, sleep=Sleeps(), breaker=breaker)
    assert seen == ["b"]
    assert result.summary == "a=circuit-open, b=200"


async def test_every_circuit_open_fails_fast_with_503_and_retry_after():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=60, clock=clock)
    breaker.record_failure("a", 503, "down")
    breaker.record_failure("b", 503, "down")
    clock.now += 15

    async def attempt(target):  # pragma: no cover - must not be called
        raise AssertionError("called an open target")

    with pytest.raises(UpstreamError) as exc:
        await execute_with_fallback(["a", "b"], attempt, max_retries_per_target=2, backoff_base=0,
                                    backoff_cap=0, sleep=Sleeps(), breaker=breaker)
    assert exc.value.status_code == 503
    assert exc.value.retry_after == 45
    assert "open circuit" in exc.value.message


async def test_circuit_opening_mid_request_stops_retrying_that_target():
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
    seen = []

    async def attempt(target):
        seen.append(target)
        if target == "a":
            raise UpstreamError(503, "down", kind=RETRY)
        return "ok"

    await execute_with_fallback(["a", "b"], attempt, max_retries_per_target=5, backoff_base=0,
                                backoff_cap=0, sleep=Sleeps(), breaker=breaker)
    assert seen == ["a", "a", "b"]  # not 6 attempts on a


async def test_caller_errors_do_not_trip_the_breaker():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=60)

    async def attempt(target):
        raise UpstreamError(400, "bad request", kind=FATAL)

    for _ in range(3):
        with pytest.raises(UpstreamError):
            await execute_with_fallback(["a"], attempt, max_retries_per_target=0, backoff_base=0,
                                        backoff_cap=0, sleep=Sleeps(), breaker=breaker)
    assert breaker.state("a") == "closed"
