"""Admin surface: hot reload, savings on the dashboard, admin token, models list."""

from __future__ import annotations

import json

import yaml

from app.cache import estimated_savings

from .conftest import CHEAP_KEY, KEY, KEYS, PRICING, ROUTING, auth, chat, sha


def reload(client, token=None):
    headers = {"X-Admin-Token": token} if token else {}
    return client.post("/admin/reload", headers=headers)


def test_reload_applies_a_new_alias_without_restart(gw, tmp_path):
    assert chat(gw, "hi", model="gpt-4").headers["x-gateway-model"] == "big"
    routing = {**ROUTING, "aliases": {**ROUTING["aliases"], "gpt-4": "small"}}
    (tmp_path / "routing.yaml").write_text(yaml.safe_dump(routing), encoding="utf-8")
    r = reload(gw)
    assert r.status_code == 200 and r.json()["status"] == "reloaded"
    assert chat(gw, "hi again", model="gpt-4").headers["x-gateway-model"] == "small"


def test_reload_adds_keys_and_prices(gw, tmp_path):
    new_key = "sk-test-added-later"
    assert chat(gw, "x", key=new_key).status_code == 401
    keys = {"keys": KEYS["keys"] + [{"name": "late", "key_hash": sha(new_key), "rpm": 5}]}
    (tmp_path / "keys.yaml").write_text(yaml.safe_dump(keys), encoding="utf-8")
    prices = {**PRICING, "big": {"input_per_1m": 1000.0, "output_per_1m": 1000.0}}
    (tmp_path / "pricing.json").write_text(json.dumps(prices), encoding="utf-8")
    assert reload(gw).status_code == 200
    assert chat(gw, "now I exist", key=new_key).status_code == 200
    usage = gw.get("/admin/usage?format=json").json()
    late = next(row for row in usage["by_key"] if row["virtual_key"] == "late")
    assert late["cost_usd"] == round(12 / 1_000_000 * 1000.0, 8)  # the new price was used


def test_reload_rejects_a_broken_config_and_keeps_serving(gw, tmp_path):
    (tmp_path / "routing.yaml").write_text(
        yaml.safe_dump({**ROUTING, "aliases": {"gpt-4": "does-not-exist"}}), encoding="utf-8")
    r = reload(gw)
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_config"
    assert any("does-not-exist" in p["message"] for p in body["problems"])
    assert chat(gw, "still alive", model="gpt-4").headers["x-gateway-model"] == "big"

    (tmp_path / "providers.yaml").write_text("providers: [broken", encoding="utf-8")
    assert reload(gw).status_code == 400
    assert chat(gw, "still alive 2", model="big").status_code == 200


def test_reload_swaps_mock_providers(make_gateway, tmp_path):
    providers = {"providers": {"m": {"type": "mock"}}, "models": {"x": {"provider": "m"}}}
    gw = make_gateway(providers=providers, routing={"fallbacks": {}}, embed_model="local/hash")
    assert chat(gw, "hello", model="x").status_code == 200
    providers["providers"]["m"]["fail_status"] = 503
    (tmp_path / "providers.yaml").write_text(yaml.safe_dump(providers), encoding="utf-8")
    assert reload(gw).status_code == 200
    r = chat(gw, "hello, again?", model="x")
    assert r.status_code == 502 and "m:x=503" in r.headers["x-gateway-attempts"]


def test_admin_token_protects_every_admin_endpoint(make_gateway):
    gw = make_gateway(admin_token="s3cret")
    for method, path in (("GET", "/admin/usage"), ("GET", "/admin/health"), ("POST", "/admin/reload")):
        assert gw.request(method, path).status_code == 401
        assert gw.request(method, path, headers={"X-Admin-Token": "wrong"}).status_code == 401
        assert gw.request(method, path, headers={"X-Admin-Token": "s3cret"}).status_code == 200


def test_usage_reports_savings_that_match_estimated_savings(gw):
    for _ in range(3):
        chat(gw, "cache me", model="big")      # 1 miss + 2 exact hits
    chat(gw, "CACHE me!", model="big")         # semantic hit
    usage = gw.get("/admin/usage?format=json").json()
    totals = usage["totals"]
    assert totals["cache_hits"] == 3
    expected = estimated_savings(3, 7, 5, PRICING["big"]["input_per_1m"], PRICING["big"]["output_per_1m"])
    assert totals["saved_usd"] == expected > 0
    assert usage["by_key"][0]["saved_usd"] == expected
    assert next(r for r in usage["by_model"] if r["model"] == "big")["saved_usd"] == expected
    assert usage["cache"]["entries"] == 1 and usage["cache"]["semantic_entries"] == 1


def test_usage_html_shows_savings_and_escapes_names(make_gateway):
    evil = "sk-test-evil"
    keys = {"keys": KEYS["keys"] + [{"name": "<script>alert(1)</script>", "key_hash": sha(evil), "rpm": 10}]}
    gw = make_gateway(keys=keys)
    chat(gw, "hello", key=evil)
    chat(gw, "hello", key=evil)
    page = gw.get("/admin/usage").text
    assert "Saved USD" in page and "saved by the cache" in page
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_models_list_requires_a_key_and_respects_allow_lists(gw):
    assert gw.get("/v1/models").status_code == 401
    everything = {m["id"] for m in gw.get("/v1/models", headers=auth(KEY)).json()["data"]}
    assert {"big", "small", "auto", "gpt-4", "text-embedding-3-small"} <= everything
    cheap = {m["id"] for m in gw.get("/v1/models", headers=auth(CHEAP_KEY)).json()["data"]}
    assert cheap == {"small", "embed", "gpt-3.5-turbo", "text-embedding-3-small"}


def test_root_and_healthz(gw):
    assert gw.get("/healthz").json()["status"] == "ok"
    root = gw.get("/").json()
    assert "/admin/reload" in root["endpoints"] and root["version"]
