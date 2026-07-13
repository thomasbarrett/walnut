"""A tiny OpenAI-compatible client used by ``walnut chat``.

Wraps ``httpx`` so the CLI can list models and request chat completions from
any OpenAI-compatible backend (including ``walnut serve``).
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx


class ChatClient:
    """Minimal client for the OpenAI-compatible chat + models endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout
        )

    def __enter__(self) -> ChatClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def list_models(self) -> list[str]:
        """Return the ids of models advertised by the backend."""
        resp = self._client.get("/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def default_model(self) -> str:
        """Return the first model the backend reports."""
        models = self.list_models()
        if not models:
            raise RuntimeError(f"No models available at {self.base_url}")
        return models[0]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
    ) -> str:
        """Request a non-streaming completion and return its text."""
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
    ) -> Iterator[str]:
        """Request a streaming completion, yielding content deltas."""
        import json

        with self._client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"]
                if content := delta.get("content"):
                    yield content
