"""Cost math and the usage ledger."""

from app.accounting import Accounting, Pricing, approx_tokens
from app.db import Database

PRICES = {
    "default": {"input_per_1m": 0.20, "output_per_1m": 0.20},
    "meta/llama-3.3-70b-instruct": {"input_per_1m": 0.90, "output_per_1m": 0.90},
    "ollama/llama3.1": {"input_per_1m": 0.0, "output_per_1m": 0.0},
}


def pricing() -> Pricing:
    return Pricing.from_dict(PRICES)


def test_cost_math():
    cost = pricing().cost("meta/llama-3.3-70b-instruct", prompt_tokens=1000, completion_tokens=500)
    expected = round(1000 / 1_000_000 * 0.9 + 500 / 1_000_000 * 0.9, 8)
    assert cost == expected


def test_unknown_model_uses_default_price():
    assert pricing().cost("mystery-model", 1_000_000, 0) == 0.20


def test_local_model_is_free():
    assert pricing().cost("ollama/llama3.1", 5000, 5000) == 0.0


def test_approx_tokens():
    assert approx_tokens("") == 1
    assert approx_tokens("x" * 40) == 10


def test_record_and_month_cost():
    acc = Accounting(Database(":memory:"), pricing())
    cost = acc.record(
        virtual_key="k1", model="meta/llama-3.3-70b-instruct", provider="nvidia",
        endpoint="chat", prompt_tokens=1000, completion_tokens=1000, cached=False,
    )
    assert cost > 0
    assert acc.month_cost("k1") == cost


def test_cached_request_costs_nothing():
    acc = Accounting(Database(":memory:"), pricing())
    live = acc.record(
        virtual_key="k1", model="meta/llama-3.3-70b-instruct", provider="nvidia",
        endpoint="chat", prompt_tokens=1000, completion_tokens=1000, cached=False,
    )
    cached = acc.record(
        virtual_key="k1", model="meta/llama-3.3-70b-instruct", provider="nvidia",
        endpoint="chat", prompt_tokens=1000, completion_tokens=1000, cached=True,
    )
    assert cached == 0.0
    assert round(acc.month_cost("k1"), 8) == round(live, 8)


def test_summary_aggregates():
    acc = Accounting(Database(":memory:"), pricing())
    acc.record(virtual_key="k1", model="meta/llama-3.3-70b-instruct", provider="nvidia",
               endpoint="chat", prompt_tokens=100, completion_tokens=100, cached=False)
    acc.record(virtual_key="k1", model="meta/llama-3.1-8b-instruct", provider="nvidia",
               endpoint="chat", prompt_tokens=50, completion_tokens=50, cached=True)
    acc.record(virtual_key="k2", model="ollama/llama3.1", provider="ollama",
               endpoint="chat", prompt_tokens=10, completion_tokens=10, cached=False)
    summary = acc.summary()
    assert summary["totals"]["requests"] == 3
    assert summary["totals"]["cache_hits"] == 1
    keys = {row["virtual_key"] for row in summary["by_key"]}
    assert keys == {"k1", "k2"}
    models = {row["model"] for row in summary["by_model"]}
    assert "meta/llama-3.3-70b-instruct" in models


def test_month_cost_isolated_by_key():
    acc = Accounting(Database(":memory:"), pricing())
    acc.record(virtual_key="k1", model="meta/llama-3.3-70b-instruct", provider="nvidia",
               endpoint="chat", prompt_tokens=1000, completion_tokens=0, cached=False)
    assert acc.month_cost("k2") == 0.0
