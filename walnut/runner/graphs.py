"""CUDA graph capture for the decode step.

A decode step launches on the order of a thousand small kernels, more than the
host can issue at the rate the GPU retires them. Capturing the step once and
replaying it turns those launches into a single call.

Capture requires every buffer the step touches to keep a fixed address, which
is what `KVCache` and `ConvState` provide — and, once the cache is paged, what
the pool's block table provides too: *which* pages a row is holding changes
under a captured graph every time a sequence retires, and the graph reads them
because it recorded a lookup into that table rather than the pages it found
there. Warm-up and capture run the step for real, so a cache holding state a
sequence still needs is snapshotted first and restored afterwards — see
``restore``, which a pool does not need and cannot afford, being the size of
every row and page at once.

A capture also fixes the batch size, which a serving batch does not hold still.
`DecodeGraphs` answers that the way vLLM does: capture a graph per power-of-two
batch size, replay the smallest one that fits, and let the spare rows compute
against pages that belong to no one.

Context length would be a second such axis, except that `Attention` decodes
through `varlen_attn`, which takes each row's length as a *tensor*. Length is
therefore data inside the graph rather than shape around it, and one capture
per batch size serves a row at any point in its sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from walnut.cache import CacheView

if TYPE_CHECKING:
    from walnut.models.protocol import Forward


class DecodeGraph:
    """A captured single-token decode step, replayable at any position.

    Build it after the prefill pass: capture records whichever branch the
    caches select, and the decode branch only exists once they hold state.

    ``batch_size`` rows are captured and every replay runs all of them; a
    caller with fewer live sequences pads (see `DecodeGraphs`).
    """

    def __init__(
        self,
        model: Forward,
        cache: CacheView,
        device: torch.device,
        batch_size: int = 1,
        warmup: int = 3,
        restore: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.restore = restore
        self.token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.position = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        # Views, so a graph over a bucket addresses only the rows and positions
        # it covers while still writing the pool every other bucket reads.
        self.cache = cache.view(0, batch_size)
        self.capture(model, warmup)

    def capture(self, model: Forward, warmup: int = 3) -> None:
        """Warm up and record the decode step into a replayable graph.

        Split out of `__init__` so a profile names it, separating capture cost
        from the prefill it follows.

        Warm-up and capture write the cache for real, which matters when it
        already holds a sequence's state — hence the snapshot. It costs a copy
        of every buffer the graph covers, so ``restore=False`` says the caller
        does not need one: `walnut.scheduler` captures against an empty pool
        and resets a row before assigning it, and cloning a whole pool per
        bucket would put peak memory at twice the cache it just allocated.
        """
        buffers = self.cache.buffers() if self.restore else []
        saved = [buffer.clone() for buffer in buffers]

        with torch.no_grad():
            # Warm up on a side stream first: this settles cuBLAS and autotune
            # workspaces, which must not be allocated during capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    model(self.token, positions=self.position, cache=self.cache)
            torch.cuda.current_stream().wait_stream(stream)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.logits = model(
                    self.token, positions=self.position, cache=self.cache
                )

        for buffer, original in zip(buffers, saved, strict=True):
            buffer.copy_(original)

    def replay(self, token: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Run the captured step for ``token`` (B, 1) at ``position`` (B, 1).

        The returned logits are the graph's own output buffer, which the next
        replay overwrites; consume them before replaying again.
        """
        self.token.copy_(token)
        self.position.copy_(position)
        self.graph.replay()
        return self.logits


def buckets(limit: int, smallest: int = 1) -> list[int]:
    """Sizes to capture for an axis running up to ``limit``.

    Powers of two, so the number of captures grows with the logarithm of the
    axis while the padding a replay carries stays under a factor of two.
    ``smallest`` puts a floor on it, for an axis whose small sizes all cost
    the same.
    """
    sizes = []
    size = min(smallest, limit)
    while size < limit:
        sizes.append(size)
        size *= 2
    sizes.append(limit)
    return sizes


class DecodeGraphs:
    """One `DecodeGraph` per batch bucket, dispatching on how many rows are live.

    Every bucket is captured up front. Capturing on demand would move the cost
    onto whichever request first pushes the batch past a bucket edge, which is
    a stall in the middle of serving rather than one at load.
    """

    def __init__(
        self,
        model: Forward,
        cache: CacheView,
        device: torch.device,
        max_batch_size: int,
        warmup: int = 3,
        restore: bool = True,
    ) -> None:
        self.graphs = {
            size: DecodeGraph(model, cache, device, size, warmup, restore)
            for size in buckets(max_batch_size)
        }

    def replay(self, token: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Run the smallest captured step covering ``token``'s rows.

        The padding rows keep the tokens and positions of whatever ran before,
        which is harmless: a row's arithmetic reads and writes only the pages
        its own block table names, and a row outside the live set has an empty
        table, which names `walnut.cache.CachePool.SCRATCH` and nothing else.
        """
        rows = token.shape[0]
        size = min(s for s in self.graphs if s >= rows)
        graph = self.graphs[size]
        graph.token[:rows].copy_(token)
        graph.position[:rows].copy_(position)
        graph.graph.replay()
        return graph.logits[:rows]
