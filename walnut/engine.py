"""The inference engine seam.

`serve` loads a model through `load_model`, which returns an
`Engine`. The server only depends on this interface, so the real
native walnut engine can be plugged in without touching the HTTP layer.

To wire up real inference, implement an `Engine` subclass (loading
weights in ``__init__`` / a classmethod and producing tokens in
`Engine.generate`) and return it from `load_model`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """A single chat message in the OpenAI schema."""

    role: str
    content: str


@dataclass
class GenerationConfig:
    """Sampling parameters for a single generation request."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] | None = None


class Engine:
    """Interface between the OpenAI-compatible server and a walnut model.

    Concrete engines carry a loaded model and turn a list of chat messages
    into generated text. Subclasses must set `model_id` and implement
    `generate`; `stream` is optional and defaults to yielding the
    full completion in one chunk.
    """

    #: Identifier reported by ``GET /v1/models`` and echoed in responses.
    model_id: str

    def generate(self, messages: list[Message], config: GenerationConfig) -> str:
        """Return a completion for ``messages``."""
        raise NotImplementedError

    def stream(
        self, messages: list[Message], config: GenerationConfig
    ) -> Iterator[str]:
        """Yield incremental completion chunks.

        The default implementation calls `generate` and yields the whole
        result once; override it for true token streaming.
        """
        yield self.generate(messages, config)


class EchoEngine(Engine):
    """Placeholder engine used until the native walnut engine is wired in.

    It performs no real inference — it just echoes the last user turn so the
    server and client can be exercised end to end. Replace it in
    `load_model` with the real engine.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, messages: list[Message], config: GenerationConfig) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        text = f"[{self.model_id} placeholder] {last_user}".strip()
        # Approximate max_tokens by trimming on whitespace so limits are visible.
        if config.max_tokens > 0:
            words = text.split()
            if len(words) > config.max_tokens:
                text = " ".join(words[: config.max_tokens])
        return text


def load_model(model: str) -> Engine:
    """Load ``model`` (a Hugging Face id or local path) into an engine.

    This is the integration point for the native walnut inference engine.
    It currently returns an `EchoEngine` placeholder so ``serve`` runs
    end to end; swap the return value for the real engine once it exists.
    """
    return EchoEngine(model_id=model)
