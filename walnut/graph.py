"""CUDA graph capture for the decode step.

A decode step launches on the order of a thousand small kernels, more than the
host can issue at the rate the GPU retires them. Capturing the step once and
replaying it turns those launches into a single call.

Capture requires every buffer the step touches to keep a fixed address, which
is what `KVCache` and `ConvState` provide. Warm-up and capture run the step for
real, so the caches are snapshotted first and restored afterwards.
"""

from __future__ import annotations

from typing import Any

import torch

from walnut.layers.attention import KVCache
from walnut.layers.cache import Cache
from walnut.layers.linear_attention import ConvState


def _buffers(cache: list[Cache]) -> list[torch.Tensor]:
    """Every mutable state tensor across a per-layer cache list."""
    tensors: list[torch.Tensor] = []
    for entry in cache:
        if isinstance(entry, KVCache):
            tensors += [entry.k, entry.v]
        elif isinstance(entry, ConvState):
            tensors += [entry.conv, entry.recurrent]
    return tensors


class DecodeGraph:
    """A captured single-token decode step, replayable at any position.

    Build it after the prefill pass: capture records whichever branch the
    caches select, and the decode branch only exists once they hold state.
    """

    def __init__(
        self,
        model: Any,
        cache: list[Cache],
        device: torch.device,
        warmup: int = 3,
    ) -> None:
        self.token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.capture(model, cache, warmup)

    def capture(self, model: Any, cache: list[Cache], warmup: int = 3) -> None:
        """Warm up and record the decode step into a replayable graph.

        Split out of `__init__` so a profile names it, separating capture cost
        from the prefill it follows.
        """
        buffers = _buffers(cache)
        saved = [buffer.clone() for buffer in buffers]

        with torch.no_grad():
            # Warm up on a side stream first: this settles cuBLAS and autotune
            # workspaces, which must not be allocated during capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    model(self.token, positions=self.position, cache=cache)
            torch.cuda.current_stream().wait_stream(stream)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.logits = model(self.token, positions=self.position, cache=cache)

        for buffer, original in zip(buffers, saved, strict=True):
            buffer.copy_(original)

    def replay(self, token: torch.Tensor, position: int) -> torch.Tensor:
        """Run the captured step for ``token`` at ``position``.

        The returned logits are the graph's own output buffer, which the next
        replay overwrites; consume them before replaying again.
        """
        self.token.copy_(token)
        self.position.fill_(position)
        self.graph.replay()
        return self.logits
