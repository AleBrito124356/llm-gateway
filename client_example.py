"""Use the gateway from the official OpenAI Python SDK.

The only change versus talking to OpenAI directly is ``base_url`` and the
``api_key`` (here a *gateway* virtual key, not an upstream provider key).

    pip install openai
    python client_example.py

Assumes the gateway is running on http://localhost:8000 (override with
GATEWAY_URL) and that keys.yaml still contains the shipped demo key. No NVIDIA
key handy? Serve the offline mock config instead - nothing else changes:

    llm-gateway serve --config-dir examples/demo
"""

from __future__ import annotations

import os

from openai import OpenAI

# The demo virtual key from the shipped keys.yaml. Replace with your own
# (mint one with: llm-gateway keys generate).
GATEWAY_KEY = os.getenv("GATEWAY_KEY", "sk-gw-demo-000-not-a-real-secret")
BASE_URL = os.getenv("GATEWAY_URL", "http://localhost:8000/v1")

client = OpenAI(base_url=BASE_URL, api_key=GATEWAY_KEY)


def chat() -> None:
    print("== chat (non-streaming) ==")
    resp = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Name three uses for an LLM gateway."},
        ],
        temperature=0.2,
    )
    print(resp.choices[0].message.content)
    print("usage:", resp.usage)


def chat_auto_routing() -> None:
    print("\n== chat with the virtual 'auto' model ==")
    # A short, plain prompt routes to the cheap model; code or long prompts route
    # to the strong model. The chosen model comes back in resp.model.
    resp = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
    )
    print("routed to:", resp.model)
    print(resp.choices[0].message.content)


def chat_streaming() -> None:
    print("\n== chat (streaming) ==")
    stream = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Write a haiku about caching."}],
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


def embeddings() -> None:
    print("\n== embeddings ==")
    resp = client.embeddings.create(
        model=os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"),
        input=["semantic caching saves money", "the cat sat on the mat"],
    )
    for item in resp.data:
        print(f"vector[{item.index}] dims={len(item.embedding)}")


if __name__ == "__main__":
    chat()
    chat_auto_routing()
    chat_streaming()
    embeddings()
    print(
        "\nRun the second time to see cache hits: watch the X-Cache header "
        "and /admin/usage."
    )
