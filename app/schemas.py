"""OpenAI-compatible request/response models.

Only the fields the gateway inspects are typed. Unknown fields are preserved
(``extra="allow"``) so the raw body can be forwarded to the upstream unchanged;
any OpenAI SDK feature you do not see named here still passes through.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def text(self) -> str:
        """Flatten content to plain text for heuristics and embeddings."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for chunk in self.content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                parts.append(str(chunk.get("text", "")))
        return "\n".join(parts)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    response_format: Optional[dict[str, Any]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    seed: Optional[int] = None
    user: Optional[str] = None

    def wants_json(self) -> bool:
        return bool(self.response_format) and self.response_format.get("type") == "json_object"


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: Union[str, list[str], list[int], list[list[int]]]
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None


def new_completion_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def build_chat_response(model: str, content: str, usage: dict[str, int]) -> dict[str, Any]:
    """Assemble an OpenAI-shaped chat completion (used for cached stream replay)."""
    return {
        "id": new_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
