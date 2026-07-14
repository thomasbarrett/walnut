"""The inference engine seam.

`serve` loads a model through `load_model`, which returns an
`Engine`. The server only depends on this interface, so the real
native walnut engine can be plugged in without touching the HTTP layer.

To wire up real inference, implement an `Engine` subclass (loading
weights in ``__init__`` / a classmethod and producing tokens in
`Engine.generate`) and return it from `load_model`.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from walnut.models import resolve_model_class
from walnut.sampler import SamplingParams


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


def _load_hf_weights(model: Any, model_id: str) -> None:
    """Download the checkpoint's safetensors and stream them into ``model``."""
    root = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights found for {model_id!r}")

    def weights() -> Iterator[tuple[str, torch.Tensor]]:
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as shard:
                for name in shard.keys():  # noqa: SIM118 (safetensors handle)
                    yield name, shard.get_tensor(name)

    model.load_weights(weights())


def _first_stop(text: str, stops: list[str] | None) -> int | None:
    """Earliest index at which any stop string occurs, else ``None``."""
    if not stops:
        return None
    hits = [i for s in stops if s and (i := text.find(s)) >= 0]
    return min(hits) if hits else None


class TorchEngine(Engine):
    """Serves a walnut PyTorch model behind the `Engine` interface.

    Builds the model from its Hugging Face config, streams the checkpoint
    weights in, and drives generation with an ``AutoTokenizer`` (chat template
    for encoding, incremental detokenization for streaming).
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        config = AutoConfig.from_pretrained(model_id)
        model: Any = resolve_model_class(config)(config)
        _load_hf_weights(model, model_id)
        self.model: Any = model.eval()
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_id)

    def _encode(self, messages: list[Message]) -> torch.Tensor:
        chat = [{"role": m.role, "content": m.content} for m in messages]
        encoded = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers may return a bare tensor or a BatchEncoding mapping.
        return encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]

    def _params(self, config: GenerationConfig) -> SamplingParams:
        return SamplingParams(
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

    def generate(self, messages: list[Message], config: GenerationConfig) -> str:
        input_ids = self._encode(messages)
        ids = list(self.model.iter_generate(input_ids, self._params(config)))
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        cut = _first_stop(text, config.stop)
        return text if cut is None else text[:cut]

    def stream(
        self, messages: list[Message], config: GenerationConfig
    ) -> Iterator[str]:
        input_ids = self._encode(messages)
        ids: list[int] = []
        emitted = ""
        for tok in self.model.iter_generate(input_ids, self._params(config)):
            ids.append(tok)
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            # Wait for complete characters (partial multi-byte decodes to U+FFFD).
            if text.endswith("�"):
                continue
            cut = _first_stop(text, config.stop)
            if cut is not None:
                if cut > len(emitted):
                    yield text[len(emitted) : cut]
                return
            if len(text) > len(emitted):
                yield text[len(emitted) :]
                emitted = text


def load_model(model: str) -> Engine:
    """Load ``model`` (a Hugging Face id or local path) into a `TorchEngine`."""
    return TorchEngine(model_id=model)
