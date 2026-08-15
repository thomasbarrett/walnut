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
import warnings
from collections.abc import Generator, Iterator
from contextlib import contextmanager
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


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a ``--device`` selection to a concrete `torch.device`.

    ``None`` and ``"auto"`` pick CUDA when it is available and CPU otherwise;
    anything else is passed through to `torch.device`. An unusable CUDA
    selection is a ``ValueError`` here rather than a failure after the load.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; install the CUDA wheels with "
                "`uv sync --extra cu130`"
            )
        count = torch.cuda.device_count()
        if resolved.index is not None and resolved.index >= count:
            raise ValueError(f"no CUDA device {resolved.index}; {count} visible")
    return resolved


def _named_dtype(name: str) -> torch.dtype:
    """Look up a floating-point `torch.dtype` by name (e.g. ``"bfloat16"``)."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"{name!r} is not a floating-point torch dtype")
    return dtype


def _config_dtype(config: Any) -> torch.dtype | None:
    """The checkpoint's declared dtype, or ``None`` if it doesn't state one."""
    # transformers>=5 renamed ``torch_dtype`` to ``dtype``.
    declared = getattr(config, "dtype", None)
    if isinstance(declared, str):
        return _named_dtype(declared)
    return declared if isinstance(declared, torch.dtype) else None


def parse_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Parse a ``--dtype`` selection without consulting a checkpoint.

    Returns ``None`` for ``None``/``"auto"`` (see `resolve_dtype`), so a
    mistyped flag can be rejected before a multi-gigabyte download.
    """
    if dtype is None or dtype == "auto":
        return None
    if isinstance(dtype, str):
        return _named_dtype(dtype)
    if not dtype.is_floating_point:
        raise ValueError(f"{dtype} is not a floating-point torch dtype")
    return dtype


def resolve_dtype(
    dtype: str | torch.dtype | None,
    config: Any,
    device: torch.device,
) -> torch.dtype:
    """Resolve a ``--dtype`` selection against the checkpoint and the device.

    ``None`` and ``"auto"`` take the dtype the checkpoint declares. A float32
    checkpoint is downcast on accelerators, where fp32 wastes both memory and
    throughput, but is left alone on CPU. bfloat16 falls back to float16 on
    CUDA devices that can't do bf16 (pre-Ampere).
    """
    resolved = parse_dtype(dtype)
    if resolved is None:  # auto: follow the checkpoint, then adjust for device
        declared = _config_dtype(config)
        resolved = declared if declared is not None else torch.float32
        if resolved == torch.float32 and device.type != "cpu":
            resolved = torch.bfloat16

    if (
        resolved == torch.bfloat16
        and device.type == "cuda"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        warnings.warn(
            f"bfloat16 is unsupported on {torch.cuda.get_device_name(device)}; "
            "using float16 instead",
            stacklevel=2,
        )
        return torch.float16
    return resolved


@contextmanager
def _build_on(device: torch.device, dtype: torch.dtype) -> Generator[None, None, None]:
    """Make module construction allocate on ``device`` in ``dtype``.

    Parameters land on the target device at the target precision, so no float32
    copy is materialized on the host. The defaults it swaps are process-global:
    one load at a time.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(previous)


def _load_hf_weights(model: Any, model_id: str, device: torch.device) -> None:
    """Download the checkpoint's safetensors and stream them into ``model``."""
    root = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights found for {model_id!r}")

    def weights() -> Iterator[tuple[str, torch.Tensor]]:
        for path in files:
            with safe_open(path, framework="pt", device=str(device)) as shard:
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

    The engine owns placement: it builds on `device` in `dtype` and encodes
    input ids there, and the KV cache and positions follow the ids.
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        cuda_graph: bool = True,
    ) -> None:
        self.model_id = model_id
        self.cuda_graph = cuda_graph
        config = AutoConfig.from_pretrained(model_id)
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, config, self.device)
        model_class = resolve_model_class(config)
        with _build_on(self.device, self.dtype):
            model: Any = model_class(config)
        _load_hf_weights(model, model_id, self.device)
        self.model: Any = model.eval()
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_id)

    def _encode(self, messages: list[Message]) -> torch.Tensor:
        chat = [{"role": m.role, "content": m.content} for m in messages]
        encoded = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers may return a bare tensor or a BatchEncoding mapping.
        if not isinstance(encoded, torch.Tensor):
            encoded = encoded["input_ids"]
        return encoded.to(self.device)

    def _params(self, config: GenerationConfig) -> SamplingParams:
        return SamplingParams(
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

    def generate(self, messages: list[Message], config: GenerationConfig) -> str:
        input_ids = self._encode(messages)
        ids = list(
            self.model.iter_generate(
                input_ids, self._params(config), cuda_graph=self.cuda_graph
            )
        )
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        cut = _first_stop(text, config.stop)
        return text if cut is None else text[:cut]

    def stream(
        self, messages: list[Message], config: GenerationConfig
    ) -> Iterator[str]:
        input_ids = self._encode(messages)
        ids: list[int] = []
        emitted = ""
        for tok in self.model.iter_generate(
            input_ids, self._params(config), cuda_graph=self.cuda_graph
        ):
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


def load_model(
    model: str,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    cuda_graph: bool = True,
) -> TorchEngine:
    """Load ``model`` (a Hugging Face id or local path) into a `TorchEngine`.

    ``device`` and ``dtype`` default to auto-selection; see `resolve_device`
    and `resolve_dtype`. ``cuda_graph`` replays decode from a captured graph
    and has no effect off CUDA.
    """
    return TorchEngine(
        model_id=model, device=device, dtype=dtype, cuda_graph=cuda_graph
    )
