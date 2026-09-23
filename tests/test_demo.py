"""The offline demo and the real OpenAI SDK against the demo config."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from app.__main__ import main as module_main
from app.config import Settings
from app.demo import DEFAULT_CONFIG_DIR, DEMO_KEY, run_demo
from app.main import build_app
from app.mock import OfflineGuardTransport


async def test_demo_runs_offline_and_every_check_passes():
    lines: list[str] = []
    report = await run_demo(out=lines.append)
    assert report.blocked == []
    failed = [name for name, ok in report.checks if not ok]
    assert failed == []
    assert len(report.checks) >= 20
    text = "\n".join(lines)
    for expected in ("HIT;EXACT", "HIT;SEMANTIC", "circuit-open", "429", "402", "0 attempted network calls"):
        assert expected in text


def test_demo_entrypoint_exit_code():
    assert module_main(["demo"]) == 0


async def test_demo_refuses_a_missing_config(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="demo config not found"):
        await run_demo(tmp_path, out=lambda _: None)


def test_openai_sdk_against_the_demo_config(tmp_path: Path):
    """The official SDK (what client_example.py uses), streaming included."""
    openai = pytest.importorskip("openai")
    guard = OfflineGuardTransport()
    settings = Settings.from_env(env={}, config_dir=str(DEFAULT_CONFIG_DIR), db_path=str(tmp_path / "sdk.db"))
    with TestClient(build_app(settings, transport=guard)) as http_client:
        base_client = getattr(openai, "_base_client", None)
        sdk_httpx = getattr(base_client, "httpx2", None) or getattr(base_client, "httpx", httpx)
        if not isinstance(http_client, sdk_httpx.Client):  # pragma: no cover - library mismatch
            pytest.skip("installed openai SDK and starlette TestClient use different httpx packages")
        client = openai.OpenAI(base_url="http://testserver/v1", api_key=DEMO_KEY, http_client=http_client)

        resp = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "system", "content": "You are concise."},
                      {"role": "user", "content": "Name three uses for an LLM gateway."}],
            temperature=0.2,
        )
        assert "Name three uses for an LLM gateway." in resp.choices[0].message.content
        assert resp.usage.total_tokens > 0

        auto = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "What is 2 + 2?"}])
        assert auto.model == "meta/llama-3.1-8b-instruct"

        stream = client.chat.completions.create(
            model="auto", stream=True, messages=[{"role": "user", "content": "Write a haiku about caching."}],
        )
        text = "".join(event.choices[0].delta.content or "" for event in stream)  # crashed before the fix
        assert "haiku about caching" in text

        emb = client.embeddings.create(model="nvidia/nv-embedqa-e5-v5",
                                       input=["semantic caching saves money", "the cat sat on the mat"])
        assert [len(item.embedding) for item in emb.data] == [384, 384]

        again = client.chat.completions.with_raw_response.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "system", "content": "You are concise."},
                      {"role": "user", "content": "Name three uses for an LLM gateway."}],
            temperature=0.2,
        )
        assert again.headers["x-cache"] == "HIT;EXACT"

        with pytest.raises(openai.AuthenticationError):
            openai.OpenAI(base_url="http://testserver/v1", api_key="sk-wrong",
                          http_client=http_client).models.list()
    assert guard.blocked == []
