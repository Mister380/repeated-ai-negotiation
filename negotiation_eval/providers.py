"""Provider adapters. Stdlib HTTP only; credentials from environment, never logged.

Each adapter returns a Completion with the text, the model id the server reports,
token usage and the provider request id, so Appendix C's archive fields are filled.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import config as C


@dataclass
class Completion:
    text: str
    returned_model: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    request_id: Optional[str]
    raw: Dict = field(default_factory=dict)


class InfraError(Exception):
    """Timeout, 5xx, 429, connection error: retryable. Not an invalid action."""


class Provider:
    def complete(self, spec: C.ModelSpec, prompt: str) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError

    def request_body(self, spec: C.ModelSpec, prompt: str) -> Dict:
        raise NotImplementedError


def _post(url: str, headers: Dict[str, str], body: Dict) -> Dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=C.REQUEST_TIMEOUT_S) as r:
            data = json.loads(r.read().decode())
            data["_request_id"] = r.headers.get("request-id") or r.headers.get("x-request-id")
            return data
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        if e.code == 429 or e.code >= 500:
            raise InfraError("HTTP {}: {}".format(e.code, detail))
        raise RuntimeError("HTTP {} (non-retryable): {}".format(e.code, detail))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise InfraError(str(e))


def _key(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError("Missing environment variable {}".format(name))
    return v


class AnthropicProvider(Provider):
    url = "https://api.anthropic.com/v1/messages"

    def request_body(self, spec, prompt):
        body = {"model": spec.api_id, "max_tokens": C.MAX_OUTPUT_TOKENS,
                "messages": [{"role": "user", "content": prompt}]}
        thinking = spec.extra.get("thinking")
        if isinstance(thinking, dict):
            body["thinking"] = thinking
        elif spec.supports_temperature:
            body["temperature"] = C.TEMPERATURE  # temperature is incompatible with thinking on Anthropic
        if "effort" in spec.extra:
            body["output_config"] = {"effort": spec.extra["effort"]}
        return body

    def complete(self, spec, prompt):
        data = _post(self.url, {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01",
                                "content-type": "application/json"}, self.request_body(spec, prompt))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        return Completion(text, data.get("model"), u.get("input_tokens"), u.get("output_tokens"),
                          data.get("id") or data.get("_request_id"), data)


class ChatCompletionsProvider(Provider):
    """OpenAI Chat Completions shape; used for OpenAI and OpenRouter (Kimi, GLM, DeepSeek, Qwen)."""

    def __init__(self, url: str, key_env: str):
        self.url, self.key_env = url, key_env

    def request_body(self, spec, prompt):
        body = {"model": spec.api_id, "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": C.MAX_OUTPUT_TOKENS}
        if spec.supports_temperature:
            body["temperature"] = C.TEMPERATURE
        if "reasoning_effort" in spec.extra:
            body["reasoning_effort"] = spec.extra["reasoning_effort"]
        if self.url.startswith("https://openrouter.ai"):
            # pin the exact model, no silent fallback routing to another model
            body["models"] = [spec.api_id]
            body["provider"] = {"allow_fallbacks": False}
        return body

    def complete(self, spec, prompt):
        data = _post(self.url, {"Authorization": "Bearer " + _key(self.key_env),
                                "Content-Type": "application/json"}, self.request_body(spec, prompt))
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        u = data.get("usage", {})
        return Completion(text, data.get("model"), u.get("prompt_tokens"), u.get("completion_tokens"),
                          data.get("id") or data.get("_request_id"), data)


class MockProvider(Provider):
    """Offline provider for tests and dry runs. `policy(spec, prompt) -> text`; may raise InfraError."""

    def __init__(self, policy: Callable[[C.ModelSpec, str], str]):
        self.policy = policy
        self.calls: List[str] = []

    def request_body(self, spec, prompt):
        return {"model": spec.api_id, "prompt": prompt}

    def complete(self, spec, prompt):
        self.calls.append(prompt)
        text = self.policy(spec, prompt)
        return Completion(text, spec.api_id, len(prompt) // 4, len(text) // 4, "mock-{}".format(len(self.calls)))


def default_providers() -> Dict[str, Provider]:
    return {
        "anthropic": AnthropicProvider(),
        "openai": ChatCompletionsProvider("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
        "openrouter": ChatCompletionsProvider("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    }
