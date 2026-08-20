"""Where decode state lives: the rows a pass runs against, and who owns them."""

from __future__ import annotations

import torch

from walnut.cache.batch import Batch
from walnut.cache.state import Cache


class CacheView:
    """The per-layer state one pass runs against, and how it is laid out.

    A pass is handed a view rather than the pool because the rows it covers are
    part of the addressing: a prefill runs against one slot and a decode against
    a bucket, and both compute cells relative to the rows they were given.
    """

    def __init__(self, caches: list[Cache], rows: int, stride: int) -> None:
        self.caches = caches
        self.rows = rows
        #: Cells one sequence spans in a token-addressed buffer. The whole of
        #: today's placement rule: sequence ``r``'s position ``p`` is cell
        #: ``r * stride + p``. A block pool replaces this field with a table
        #: and `batch` with a lookup into it; nothing else moves.
        self.stride = stride
        self._base = self._row_base()

    def _row_base(self) -> torch.Tensor | None:
        """``(rows, 1)`` first-cell-of-each-row, or None with nothing to place.

        Built once per view rather than per pass: a captured graph replays the
        addresses it recorded, so anything a pass reads has to outlive capture,
        and a view is built once where a pass runs every step.
        """
        buffers = self.buffers()
        if not buffers:
            return None
        device = buffers[0].device
        return torch.arange(self.rows, device=device).unsqueeze(1) * self.stride

    def __len__(self) -> int:
        return len(self.caches)

    def __getitem__(self, index: int) -> Cache:
        return self.caches[index]

    def __iter__(self):
        return iter(self.caches)

    def buffers(self) -> list[torch.Tensor]:
        """Every mutable tensor across every layer this view covers."""
        return [tensor for cache in self.caches for tensor in cache.buffers()]

    def view(self, start: int, stop: int) -> CacheView:
        """Rows ``[start, stop)`` of every layer, as one view."""
        return CacheView(
            [cache.view(start, stop) for cache in self.caches],
            stop - start,
            self.stride,
        )

    def slot(self, index: int) -> CacheView:
        """A single-row view, what a batch-1 prefill runs against."""
        return self.view(index, index + 1)

    def batch(self, positions: torch.Tensor) -> Batch:
        """Resolve ``positions`` against this view's placement.

        ``positions`` is what the caller has: ``(width,)`` for a pass whose rows
        all sit at the same positions, which is a lone prefill; ``(rows, width)``
        for a batch of independent sequences; ``(3, rows, width)`` for M-RoPE,
        whose first axis is the temporal one and therefore the ordinal the cache
        wants.
        """
        place = positions[0] if positions.ndim == 3 else positions
        if place.ndim == 1:
            place = place.expand(self.rows, -1)
        # A row attends through its own last position, so that position plus
        # one *is* its length.
        lengths = (place[:, -1] + 1).to(torch.int32)
        cells = place if self._base is None else self._base + place
        return Batch(place, lengths, cells)


class CachePool(CacheView):
    """Every layer's decode state, and which slots are free.

    A pool is the view of every row, plus the right to hand rows out. The pool
    is preallocated: ``max_batch_size`` slots of ``max_seq_len`` tokens each,
    taken once at start. Allocation is therefore a free list, and a request
    that does not fit is refused rather than allowed to displace a running one.
    Both of those are properties of *this* allocator and not of the interface —
    `reserve` and `release` are where a block allocator arrives, with a prefix
    tree above it handing back placements that overlap.
    """

    def __init__(
        self, caches: list[Cache], max_batch_size: int, max_seq_len: int
    ) -> None:
        super().__init__(caches, max_batch_size, max_seq_len)
        self._free = list(range(max_batch_size))

    def reserve(self) -> int | None:
        """Take a slot for a new sequence, or None if the pool is full.

        Lowest free slot first, which keeps the live set dense — a decode step
        runs whole buckets, so a sequence parked in a high slot costs every
        step the rows beneath it.
        """
        return self._free.pop(0) if self._free else None

    def release(self, slot: int) -> None:
        """Give a finished sequence's slot back."""
        self._free.append(slot)
        self._free.sort()

    @property
    def free(self) -> tuple[int, ...]:
        """Slots a new sequence could take, lowest first."""
        return tuple(self._free)

    def reset(self, slot: int) -> None:
        """Clear ``slot`` in every layer, before a new sequence takes it."""
        for cache in self.caches:
            cache.reset(slot)

    def prime(self) -> None:
        """Tell every layer its buffers count as state.

        The pool is decoded against from the first step: its slots hold zeros,
        which is what "no context yet" means, and without this a mixer that
        branches on having state would take the prefill branch on a pool that
        is merely empty.
        """
        for cache in self.caches:
            cache.prime()

    def carried(self, slot: int) -> list[torch.Tensor]:
        """``slot``'s state that a decode step advances rather than indexes."""
        return [tensor for cache in self.caches for tensor in cache.carried(slot)]
