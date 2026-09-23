"""The llm-gateway CLI, the keys CLI and import-time behaviour of app.main."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from app.cli import load_env_file, main
from app.keys import KeyStore, _cli as keys_cli, hash_key

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def shipped(tmp_path: Path) -> Path:
    for name in ("providers.yaml", "routing.yaml", "keys.yaml", "pricing.json"):
        shutil.copy(REPO / name, tmp_path / name)
    return tmp_path


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("llm-gateway 0.")


def test_check_config_ok_on_the_shipped_config(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env
    assert main(["check-config", "--config-dir", str(REPO)]) == 0
    out = capsys.readouterr().out
    assert "OK: 0 error(s)" in out


def test_check_config_fails_and_lists_every_problem(shipped, capsys, monkeypatch):
    monkeypatch.chdir(shipped)
    routing = yaml.safe_load((shipped / "routing.yaml").read_text(encoding="utf-8"))
    routing["fallbacks"]["meta/llama-3.3-70b-instruct"].append("ollama/llama9")
    routing["aliases"]["gpt-5"] = "meta/llama-9-instruct"
    (shipped / "routing.yaml").write_text(yaml.safe_dump(routing), encoding="utf-8")
    keys = (shipped / "keys.yaml").read_text(encoding="utf-8").replace(
        "db08fd09bef90acab8dd92fb2e3ee762d65248e412310a2b04a95deb1e970603", "db08fd09-not-hex")
    (shipped / "keys.yaml").write_text(keys, encoding="utf-8")

    assert main(["check-config", "--config-dir", str(shipped)]) == 1
    out = capsys.readouterr().out
    assert "fallback 'meta/llama-3.3-70b-instruct' -> 'ollama/llama9' is not a model" in out
    assert "alias 'gpt-5' -> 'meta/llama-9-instruct' is not a model" in out
    assert "key 'demo': key_hash must be 64 hex characters" in out
    assert "FAILED: 3 error(s)" in out

    assert main(["check-config", "--config-dir", str(shipped), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and report["errors"] == 3


def test_check_config_strict_fails_on_warnings(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert main(["check-config", "--config-dir", str(REPO / "examples" / "demo"), "--strict"]) == 1
    assert "(--strict)" in capsys.readouterr().out


def test_check_config_reads_dot_env(shipped, capsys, monkeypatch):
    monkeypatch.chdir(shipped)
    for name in ("NVIDIA_API_KEY", "OPENAI_API_KEY", "TOGETHER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (shipped / ".env").write_text("NVIDIA_API_KEY=nvapi-test\n", encoding="utf-8")
    assert main(["check-config", "--config-dir", str(shipped)]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA_API_KEY" not in out and "OPENAI_API_KEY" in out
    assert main(["check-config", "--env-file", "missing.env"]) == 2


def test_load_env_file_rules(tmp_path, monkeypatch):
    env = tmp_path / "x.env"
    env.write_text(
        "# comment\n\nexport GW_A=1\nGW_B = 'two words'\nGW_C=\"quoted # not a comment\"\n"
        "GW_D=value # trailing comment\nGW_KEEP=from-file\nnot a pair\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GW_KEEP", "from-env")
    for name in ("GW_A", "GW_B", "GW_C", "GW_D"):
        monkeypatch.delenv(name, raising=False)
    assert load_env_file(str(env)) == ["GW_A", "GW_B", "GW_C", "GW_D"]
    assert os.environ["GW_A"] == "1"
    assert os.environ["GW_B"] == "two words"
    assert os.environ["GW_C"] == "quoted # not a comment"
    assert os.environ["GW_D"] == "value"
    assert os.environ["GW_KEEP"] == "from-env"  # never overrides the real environment


def test_serve_launches_the_factory(shipped, monkeypatch, capsys):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.chdir(shipped)
    monkeypatch.delenv("CONFIG_DIR", raising=False)
    assert main(["serve", "--config-dir", str(shipped), "--port", "18999"]) == 0
    (target,), kwargs = calls[0]
    assert target == "app.main:create_app"
    assert kwargs["factory"] is True and kwargs["port"] == 18999 and kwargs["host"] == "127.0.0.1"
    assert os.environ["CONFIG_DIR"] == str(shipped)


def test_serve_refuses_a_missing_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_DIR", raising=False)
    assert main(["serve", "--config-dir", str(tmp_path / "nowhere")]) == 2
    assert "providers.yaml not found" in capsys.readouterr().err
    assert main(["serve", "--env-file", "nope.env"]) == 2


def test_serve_strict_refuses_config_errors(shipped, monkeypatch, capsys):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("must not start"))
    monkeypatch.chdir(shipped)
    (shipped / "routing.yaml").write_text("aliases: {gpt-5: nowhere}\n", encoding="utf-8")
    assert main(["serve", "--config-dir", str(shipped), "--strict"]) == 1
    assert "alias 'gpt-5' -> 'nowhere'" in capsys.readouterr().out


def test_keys_generate_round_trips_through_keystore(capsys):
    assert main(["keys", "generate", "--name", "mobile", "--rpm", "7", "--budget", "2.5",
                 "--models", "small", "embed*"]) == 0
    out = capsys.readouterr().out
    plaintext = next(line for line in out.splitlines() if line.startswith("sk-gw-"))
    entry = out.split("keys.yaml:\n", 1)[1]
    store = KeyStore.from_dict(yaml.safe_load("keys:\n" + entry))
    key = store.authenticate(plaintext)
    assert key is not None and key.name == "mobile" and key.rpm == 7 and key.monthly_budget_usd == 2.5
    assert key.allows_model("small") and key.allows_model("embed-x") and not key.allows_model("big")


def test_keys_hash_and_bad_rpm(capsys):
    assert main(["keys", "hash", "sk-gw-demo-000-not-a-real-secret"]) == 0
    assert capsys.readouterr().out.strip() == hash_key("sk-gw-demo-000-not-a-real-secret")
    assert main(["keys", "generate", "--rpm", "0"]) == 2
    assert keys_cli(["hash", "abc"]) == 0  # python -m app.keys still works
    assert capsys.readouterr().out.strip().endswith(hash_key("abc"))


def test_demo_command(capsys):
    assert main(["demo"]) == 0
    assert "checks passed; 0 attempted network calls" in capsys.readouterr().out


def _run(code: str, cwd: Path, **env: str) -> subprocess.CompletedProcess:
    full_env = {**os.environ, "PYTHONPATH": str(REPO), **env}
    full_env.pop("CONFIG_DIR", None)
    full_env.update(env)
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=full_env,
                          capture_output=True, text=True, timeout=60)


def test_importing_app_main_has_no_side_effects(tmp_path):
    result = _run("import app.main", tmp_path)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []  # no gateway.db, nothing read


def test_module_level_app_is_built_lazily_with_a_clear_error(tmp_path):
    result = _run("import app.main as m; m.app", tmp_path)
    assert result.returncode != 0
    assert "providers file not found" in result.stderr and "CONFIG_DIR" in result.stderr
    ok = _run("import app.main as m; print(type(m.app).__name__, m.app is m.app)", tmp_path,
              CONFIG_DIR=str(REPO / "examples" / "demo"), GATEWAY_DB=str(tmp_path / "x.db"))
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.strip() == "FastAPI True"


def test_unknown_module_attribute_still_raises():
    import app.main as m

    with pytest.raises(AttributeError):
        m.definitely_not_here  # noqa: B018
