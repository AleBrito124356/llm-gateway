"""Retry, backoff and cross-target failover order (no network, no real clock)."""

import pytest

from app.fallback import UpstreamError, backoff_delay, execute_with_fallback


async def _no_sleep(_seconds: float) -> None:
    return None


async def test_first_target_success():
    calls = []

    async def attempt(target):
        calls.append(target)
        return f"ok-{target}"

    result = await execute_with_fallback(
        ["a", "b"], attempt, max_retries_per_target=2,
        backoff_base=0, backoff_cap=0, sleep=_no_sleep,
    )
    assert result.value == "ok-a"
    assert calls == ["a"]  # second target never touched


async def test_retry_then_success_on_same_target():
    counter = {"n": 0}

    async def attempt(target):
        counter["n"] += 1
        if counter["n"] < 2:
            raise UpstreamError(503, "busy", retryable=True)
        return "ok"

    result = await execute_with_fallback(
        ["a"], attempt, max_retries_per_target=3,
        backoff_base=0, backoff_cap=0, sleep=_no_sleep,
    )
    assert result.value == "ok"
    assert counter["n"] == 2


async def test_failover_order_after_exhausting_retries():
    seen = []

    async def attempt(target):
        seen.append(target)
        if target == "a":
            raise UpstreamError(500, "down", retryable=True)
        return "ok-b"

    result = await execute_with_fallback(
        ["a", "b"], attempt, max_retries_per_target=1,
        backoff_base=0, backoff_cap=0, sleep=_no_sleep,
    )
    assert result.value == "ok-b"
    # 'a' attempted twice (initial + 1 retry), then failover to 'b'.
    assert seen == ["a", "a", "b"]


async def test_non_retryable_aborts_immediately():
    seen = []

    async def attempt(target):
        seen.append(target)
        raise UpstreamError(400, "bad request", retryable=False)

    with pytest.raises(UpstreamError) as exc:
        await execute_with_fallback(
            ["a", "b"], attempt, max_retries_per_target=2,
            backoff_base=0, backoff_cap=0, sleep=_no_sleep,
        )
    assert exc.value.status_code == 400
    assert seen == ["a"]  # no retry, no failover


async def test_all_targets_fail_raises_aggregated():
    async def attempt(target):
        raise UpstreamError(500, "boom", retryable=True)

    with pytest.raises(UpstreamError) as exc:
        await execute_with_fallback(
            ["a", "b"], attempt, max_retries_per_target=0,
            backoff_base=0, backoff_cap=0, sleep=_no_sleep,
        )
    assert "all targets failed" in exc.value.message


def test_backoff_is_exponential_and_capped():
    assert backoff_delay(0, base=0.5, cap=8) == 0.5
    assert backoff_delay(1, base=0.5, cap=8) == 1.0
    assert backoff_delay(2, base=0.5, cap=8) == 2.0
    assert backoff_delay(10, base=0.5, cap=8) == 8.0  # capped


async def test_empty_targets_raises():
    async def attempt(target):
        return "x"

    with pytest.raises(ValueError):
        await execute_with_fallback(
            [], attempt, max_retries_per_target=1,
            backoff_base=0, backoff_cap=0, sleep=_no_sleep,
        )
