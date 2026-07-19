"""Token-bucket rate limiting and monthly budget guard."""

from app.limits import RateLimiter, TokenBucket, check_budget


def test_bucket_consumes_until_empty():
    bucket = TokenBucket(capacity=2, refill_per_sec=0, tokens=2, updated=0.0)
    assert bucket.try_consume(1, now=0.0)
    assert bucket.try_consume(1, now=0.0)
    assert not bucket.try_consume(1, now=0.0)


def test_bucket_refills_over_time():
    bucket = TokenBucket(capacity=2, refill_per_sec=1, tokens=0, updated=0.0)
    assert not bucket.try_consume(1, now=0.0)
    assert bucket.try_consume(1, now=1.0)  # one token refilled after 1s
    assert not bucket.try_consume(1, now=1.0)


def test_bucket_refill_capped_at_capacity():
    bucket = TokenBucket(capacity=2, refill_per_sec=1, tokens=0, updated=0.0)
    # 100s elapsed would add 100 tokens, but capacity caps it at 2.
    assert bucket.try_consume(2, now=100.0)
    assert not bucket.try_consume(1, now=100.0)


def test_seconds_until_available():
    bucket = TokenBucket(capacity=5, refill_per_sec=1, tokens=0, updated=0.0)
    assert bucket.seconds_until(1, now=0.0) == 1.0


def test_rate_limiter_blocks_after_burst():
    limiter = RateLimiter()
    first = limiter.check("key-a", rpm=2)
    assert first.allowed and first.remaining == 1
    second = limiter.check("key-a", rpm=2)
    assert second.allowed and second.remaining == 0
    third = limiter.check("key-a", rpm=2)
    assert not third.allowed
    headers = third.headers()
    assert headers["X-RateLimit-Limit"] == "2"
    assert headers["X-RateLimit-Remaining"] == "0"
    assert "X-RateLimit-Reset" in headers


def test_rate_limiter_isolates_keys():
    limiter = RateLimiter()
    limiter.check("key-a", rpm=1)
    assert not limiter.check("key-a", rpm=1).allowed
    # A different key has its own bucket.
    assert limiter.check("key-b", rpm=1).allowed


def test_budget_allows_under_limit():
    decision = check_budget(spent_usd=0.5, monthly_budget_usd=1.0)
    assert decision.allowed
    assert decision.remaining_usd == 0.5
    assert decision.headers()["X-Budget-Remaining"] == "0.5000"


def test_budget_blocks_at_or_over_limit():
    assert not check_budget(1.0, 1.0).allowed
    assert not check_budget(1.5, 1.0).allowed


def test_budget_none_is_unlimited():
    decision = check_budget(999.0, None)
    assert decision.allowed
