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
from walnut.scheduler import Request, Scheduler


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

        Called before the response is committed, so an implementation that can
        reject a request should do it here rather than on the first pull — by
        then the status code has been sent. See `TorchEngine.stream`.
        """
        yield self.generate(messages, config)

    def close(self) -> None:
        """Release whatever the engine is holding. Default: nothing to do."""


#: Compute capability the attention kernels are built for. Below it,
#: `walnut.layers.attention` has no kernel and the load is refused rather than
#: allowed to fail inside the first request.
MIN_CUDA_CAPABILITY = (8, 0)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a ``--device`` selection to a concrete `torch.device`.

    ``None`` and ``"auto"`` pick the default CUDA device; anything else is
    passed through to `torch.device`. walnut decodes through FlashAttention's
    variable-length kernel, which exists only for CUDA and only from Ampere, so
    an unusable selection is a ``ValueError`` here rather than a failure after
    a multi-gigabyte load.
    """
    if device is None or device == "auto":
        device = "cuda"
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(
            f"walnut runs on CUDA; {resolved.type!r} is unsupported. Its "
            "attention kernel has no CPU build."
        )
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable; install the CUDA wheels with `uv sync --extra cu130`"
        )
    count = torch.cuda.device_count()
    if resolved.index is not None and resolved.index >= count:
        raise ValueError(f"no CUDA device {resolved.index}; {count} visible")
    capability = torch.cuda.get_device_capability(resolved)
    if capability < MIN_CUDA_CAPABILITY:
        name = torch.cuda.get_device_name(resolved)
        raise ValueError(
            f"{name} is compute capability {capability[0]}.{capability[1]}; "
            f"walnut needs {MIN_CUDA_CAPABILITY[0]}.{MIN_CUDA_CAPABILITY[1]} "
            "(Ampere) or newer for its attention kernel"
        )
    return resolved


#: Precisions the attention kernel has a build for, and so the only ones a
#: model can be served in. A checkpoint declaring anything else is downcast;
#: a *flag* naming anything else is an error, because the caller asked for
#: something specific and would not otherwise be told they did not get it.
SERVING_DTYPES = (torch.float16, torch.bfloat16)


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
    mistyped flag — or one naming a precision walnut cannot serve in — is
    rejected before a multi-gigabyte download.
    """
    if dtype is None or dtype == "auto":
        return None
    if isinstance(dtype, str):
        dtype = _named_dtype(dtype)
    if not dtype.is_floating_point:
        raise ValueError(f"{dtype} is not a floating-point torch dtype")
    if dtype not in SERVING_DTYPES:
        names = ", ".join(str(d).removeprefix("torch.") for d in SERVING_DTYPES)
        raise ValueError(
            f"{str(dtype).removeprefix('torch.')} is unsupported; walnut's "
            f"attention kernel is built for {names}"
        )
    return dtype


def resolve_dtype(
    dtype: str | torch.dtype | None,
    config: Any,
    device: torch.device,
) -> torch.dtype:
    """Resolve a ``--dtype`` selection against the checkpoint and the device.

    ``None`` and ``"auto"`` take the dtype the checkpoint declares. float32 is
    downcast to bfloat16: it wastes memory and throughput, and the attention
    kernel has no float32 build, so it is not a precision walnut can serve in.
    """
    resolved = parse_dtype(dtype)
    if resolved is None:  # auto: follow the checkpoint
        declared = _config_dtype(config)
        resolved = declared if declared is not None else torch.float32

    if resolved not in SERVING_DTYPES:  # only reachable from the checkpoint
        resolved = torch.bfloat16

    if resolved == torch.bfloat16 and not torch.cuda.is_bf16_supported():
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


#: Context length to preallocate the KV pool for when nothing says otherwise.
#: The pool costs ``max_batch_size * max_seq_len`` tokens of KV whether or not
#: any request is that long, and a checkpoint's own limit is often six figures,
#: so the default is a serving-shaped one rather than the model's ceiling.
DEFAULT_MAX_SEQ_LEN = 8192


def resolve_max_seq_len(max_seq_len: int | None, config: Any) -> int:
    """Context length per slot: the request, else the checkpoint's own limit
    capped at `DEFAULT_MAX_SEQ_LEN`."""
    if max_seq_len is not None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")
        return max_seq_len
    text = getattr(config, "text_config", config)
    declared = getattr(text, "max_position_embeddings", None)
    if not isinstance(declared, int):
        return DEFAULT_MAX_SEQ_LEN
    return min(declared, DEFAULT_MAX_SEQ_LEN)


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

    Requests do not run here: they are handed to a `Scheduler`, which decodes
    up to ``max_batch_size`` of them as one batch. `generate` and `stream`
    block on their own request's tokens, so the HTTP layer stays one thread per
    request while the GPU sees one batch.
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        cuda_graph: bool = True,
        compile: bool = True,
        autotune: bool = True,
        max_batch_size: int = 8,
        max_seq_len: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.cuda_graph = cuda_graph
        self.compile = compile
        self.autotune = autotune
        config = AutoConfig.from_pretrained(model_id)
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, config, self.device)
        self.max_batch_size = max_batch_size
        self.max_seq_len = resolve_max_seq_len(max_seq_len, config)
        model_class = resolve_model_class(config)
        with _build_on(self.device, self.dtype):
            model: Any = model_class(config)
        _load_hf_weights(model, model_id, self.device)
        self.model: Any = model.eval()
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_id)
        # Allocating the pool here rather than on the first request turns a
        # cache that does not fit into a failure at load, where it is legible.
        self.scheduler = Scheduler(
            self.model,
            self.device,
            max_batch_size=max_batch_size,
            max_seq_len=self.max_seq_len,
            cuda_graph=cuda_graph,
            compile=compile,
            autotune=autotune,
        )

    def start(self) -> None:
        """Compile, capture the decode graphs, and start the scheduler.

        Optional: the first request does this itself. Calling it before the
        server accepts traffic keeps that one-off cost out of a request.
        """
        self.scheduler.start()

    def close(self) -> None:
        """Stop the scheduler once the running batch drains."""
        self.scheduler.close()

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

    def _submit(self, messages: list[Message], config: GenerationConfig) -> Request:
        """Hand one request to the scheduler; its tokens arrive on its queue."""
        params = self._params(config)
        stop_ids = frozenset(params.stop_token_ids)
        if not stop_ids and self.model.eos_token_id is not None:
            stop_ids = frozenset({self.model.eos_token_id})
        return self.scheduler.submit(
            Request(prompt=self._encode(messages), params=params, stop_ids=stop_ids)
        )

    def generate(self, messages: list[Message], config: GenerationConfig) -> str:
        ids = list(self._submit(messages, config).stream())
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        cut = _first_stop(text, config.stop)
        return text if cut is None else text[:cut]

    def stream(
        self, messages: list[Message], config: GenerationConfig
    ) -> Iterator[str]:
        """Submit now, yield later.

        Deliberately not a generator: the submission — and so the rejection of
        a request that does not fit — has to happen when the caller asks, not
        when it first pulls. A server that only finds out on the first pull has
        already sent its response headers and can no longer answer with a
        status code. Submitting costs nothing to wait for; the request is
        queued, and the model runs on the scheduler's thread.
        """
        return self._pieces(self._submit(messages, config), config)

    def _pieces(self, request: Request, config: GenerationConfig) -> Iterator[str]:
        """Detokenize a request's ids into the text deltas a client sees."""
        ids: list[int] = []
        emitted = ""
        for tok in request.stream():
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
    compile: bool = True,
    autotune: bool = True,
    max_batch_size: int = 8,
    max_seq_len: int | None = None,
) -> TorchEngine:
    """Load ``model`` (a Hugging Face id or local path) into a `TorchEngine`.

    ``device`` and ``dtype`` default to auto-selection; see `resolve_device`
    and `resolve_dtype`. ``cuda_graph`` replays decode from a captured graph
    and has no effect off CUDA. ``compile`` runs the decode step through
    `torch.compile`, paying a one-off compile for fused kernels. ``autotune``
    has that compile benchmark a Triton template per projection rather than
    take cuBLAS on faith; it is ignored without ``compile``.

    ``max_batch_size`` is how many requests the scheduler decodes as one batch,
    and ``max_seq_len`` the context each of its slots is preallocated for; see
    `walnut.scheduler.Scheduler` and `resolve_max_seq_len`.
    """
    return TorchEngine(
        model_id=model,
        device=device,
        dtype=dtype,
        cuda_graph=cuda_graph,
        compile=compile,
        autotune=autotune,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
