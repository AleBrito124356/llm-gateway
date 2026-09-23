"""Virtual keys: hashing, lookup, allow-lists and config coercion."""

from __future__ import annotations

from app.keys import KeyStore, VirtualKey, generate_key, hash_key


def test_generate_and_hash():
    key = generate_key()
    assert key.startswith("sk-gw-") and len(key) > 30
    assert generate_key() != key
    assert hash_key("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_keystore_authenticates_by_hash_only():
    store = KeyStore.from_dict({"keys": [{"name": "a", "key_hash": hash_key("sk-a"), "rpm": 5,
                                          "monthly_budget_usd": 3}]})
    key = store.authenticate("sk-a")
    assert key is not None and key.name == "a" and key.rpm == 5 and key.monthly_budget_usd == 3.0
    assert store.authenticate("sk-b") is None
    assert store.authenticate("") is None
    assert len(store) == 1


def test_hashes_are_normalised():
    upper = hash_key("sk-up").upper()
    store = KeyStore.from_dict({"keys": [{"name": "u", "key_hash": f"  {upper} "}]})
    assert store.authenticate("sk-up") is not None
    # An unquoted all-digit YAML value arrives as an int; it must not crash.
    assert len(KeyStore.from_dict({"keys": [{"name": "z", "key_hash": 0}]})) == 1


def test_glob_allow_lists():
    key = VirtualKey(name="k", key_hash="", allowed_models=["meta/*", "nvidia/nv-embedqa-e5-v5"])
    assert key.allows_model("meta/llama-3.1-8b-instruct")
    assert key.allows_model("nvidia/nv-embedqa-e5-v5")
    assert not key.allows_model("gpt-4o-mini")
    assert VirtualKey(name="all", key_hash="").allows_model("anything")


def test_missing_or_empty_keys_file(tmp_path):
    assert len(KeyStore.from_file(str(tmp_path / "nope.yaml"))) == 0
    empty = tmp_path / "keys.yaml"
    empty.write_text("", encoding="utf-8")
    assert len(KeyStore.from_file(str(empty))) == 0
