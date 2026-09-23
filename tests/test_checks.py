"""check-config: cross-reference validation of the four config files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from app.checks import ERROR, WARNING, validate_config
from app.config import Settings

REPO = Path(__file__).resolve().parent.parent


def settings_for(directory: Path, **overrides) -> Settings:
    return Settings.from_env(env={}, config_dir=str(directory), **overrides)


def messages(problems, level=None) -> list[str]:
    return [f"{p.file}: {p.message}" for p in problems if level is None or p.level == level]


@pytest.fixture
def shipped(tmp_path: Path) -> Path:
    for name in ("providers.yaml", "routing.yaml", "keys.yaml", "pricing.json"):
        shutil.copy(REPO / name, tmp_path / name)
    return tmp_path


def edit_yaml(path: Path, change) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_shipped_config_has_no_errors():
    problems = validate_config(settings_for(REPO), env={"NVIDIA_API_KEY": "x", "OPENAI_API_KEY": "x",
                                                        "TOGETHER_API_KEY": "x"})
    assert messages(problems, ERROR) == []
    # The only thing left to flag is the published demo key.
    assert messages(problems, WARNING) == [
        "keys.yaml: key 'demo' uses the public demo key from the README; rotate it before real use"
    ]


def test_demo_config_has_no_errors():
    assert messages(validate_config(settings_for(REPO / "examples" / "demo"), env={}), ERROR) == []


def test_unset_provider_keys_are_warnings(shipped):
    warnings = messages(validate_config(settings_for(shipped), env={}), WARNING)
    assert any("NVIDIA_API_KEY, which is not set" in w for w in warnings)
    assert not any("OLLAMA_API_KEY" in w for w in warnings)  # has a default_api_key


def test_broken_cross_references_are_all_reported(shipped):
    edit_yaml(shipped / "routing.yaml", lambda d: (
        d["fallbacks"]["meta/llama-3.3-70b-instruct"].append("ollama/llama9"),
        d["aliases"].__setitem__("gpt-5", "meta/llama-9-instruct"),
        d["auto"]["models"].__setitem__("cheap", "tiny/model"),
        d["auto"]["rules"].append({"if": "sunny", "use": "strong"}),
        d["fallbacks"].__setitem__("ghost/model", ["ollama/llama3.1"]),
    ))
    edit_yaml(shipped / "keys.yaml", lambda d: d["keys"][0].__setitem__("key_hash", "not-a-hash"))
    errors = messages(validate_config(settings_for(shipped, embed_model="nope/embed"), env={}), ERROR)
    assert "routing.yaml: fallback 'meta/llama-3.3-70b-instruct' -> 'ollama/llama9' is not a model in providers.yaml" in errors
    assert "routing.yaml: alias 'gpt-5' -> 'meta/llama-9-instruct' is not a model in providers.yaml" in errors
    assert "routing.yaml: auto.models.cheap 'tiny/model' is not a model in providers.yaml" in errors
    assert "routing.yaml: fallbacks key 'ghost/model' is not a model in providers.yaml" in errors
    assert any("auto.rules[4]" in e for e in errors)
    assert any("key 'demo': key_hash must be 64 hex characters" in e for e in errors)
    assert "EMBED_MODEL: 'nope/embed' is not a model in providers.yaml (use one that is, or local/hash)" in errors
    assert len(errors) == 7


def test_key_file_problems(shipped):
    keys = {"keys": [
        {"name": "a", "key_hash": "a" * 64, "rpm": 10},
        {"name": "a", "key_hash": "b" * 64},
        {"name": "c", "key_hash": "a" * 64},
        {"name": "d", "key_hash": 12345, "rpm": 0},
        {"name": "e", "key_hash": "C" * 64, "monthly_budget_usd": -1, "allowed_models": ["nothing/*"]},
        {"key_hash": "d" * 64},
        "not a mapping",
        {"name": "f", "key_hash": "f" * 64, "allowed_models": "meta/*"},
    ]}
    (shipped / "keys.yaml").write_text(yaml.safe_dump(keys), encoding="utf-8")
    problems = validate_config(settings_for(shipped), env={})
    errors, warnings = messages(problems, ERROR), messages(problems, WARNING)
    assert "keys.yaml: duplicate key name 'a' (usage and limits would merge)" in errors
    assert "keys.yaml: key 'c': key_hash is used by another key" in errors
    assert "keys.yaml: key 'd': key_hash was read as a int (12345); quote it in YAML" in errors
    assert "keys.yaml: key 'd': rpm must be a positive integer, got 0" in errors
    assert "keys.yaml: key 'e': monthly_budget_usd must be >= 0, got -1" in errors
    assert "keys.yaml: keys[5] has no name" in errors
    assert "keys.yaml: keys[6] must be a mapping" in errors
    assert "keys.yaml: key 'f': allowed_models must be a list of strings" in errors
    assert "keys.yaml: key 'e': allowed_models 'nothing/*' matches no model" in warnings


def test_unquoted_all_digit_hash_in_yaml_is_caught(shipped):
    (shipped / "keys.yaml").write_text("keys:\n  - name: z\n    key_hash: " + "0" * 64 + "\n", encoding="utf-8")
    errors = messages(validate_config(settings_for(shipped), env={}), ERROR)
    assert errors == ["keys.yaml: key 'z': key_hash was read as a int (0); quote it in YAML"]


def test_pricing_problems(shipped):
    prices = json.loads((shipped / "pricing.json").read_text(encoding="utf-8"))
    del prices["gpt-4o-mini"]
    prices["meta/llama-3.1-8b-instruct"] = {"input_per_1m": -1, "output_per_1m": "cheap"}
    prices["broken"] = 3
    (shipped / "pricing.json").write_text(json.dumps(prices), encoding="utf-8")
    problems = validate_config(settings_for(shipped), env={})
    assert "pricing.json: no price for 'gpt-4o-mini': it is billed at the 'default' price" in messages(problems, WARNING)
    errors = messages(problems, ERROR)
    assert "pricing.json: 'meta/llama-3.1-8b-instruct'.input_per_1m must be a number >= 0" in errors
    assert "pricing.json: 'meta/llama-3.1-8b-instruct'.output_per_1m must be a number >= 0" in errors
    assert "pricing.json: 'broken' must be an object with input_per_1m/output_per_1m" in errors


def test_missing_and_unparseable_files(tmp_path):
    (tmp_path / "providers.yaml").write_text("providers: [unclosed", encoding="utf-8")
    (tmp_path / "pricing.json").write_text("{not json", encoding="utf-8")
    problems = validate_config(settings_for(tmp_path), env={})
    errors = messages(problems, ERROR)
    assert any(e.startswith("providers.yaml: cannot be parsed") for e in errors)
    assert any(e.startswith("routing.yaml: file not found") for e in errors)
    assert any(e.startswith("pricing.json: cannot be parsed") for e in errors)
    assert "keys.yaml: no keys file: every request will get 401 (REQUIRE_AUTH=true)" in messages(problems, WARNING)


def test_structural_provider_errors_and_local_hash(tmp_path):
    (tmp_path / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"p": {"type": "mock"}},
        "models": {"m": {"provider": "missing"}},
    }), encoding="utf-8")
    (tmp_path / "routing.yaml").write_text("aliases: {}\n", encoding="utf-8")
    problems = validate_config(settings_for(tmp_path, embed_model="local/hash"), env={})
    assert "providers.yaml: model 'm' references unknown provider 'missing'" in messages(problems, ERROR)
    assert not any("EMBED_MODEL" in m for m in messages(problems))
