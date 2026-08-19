"""Per-layer decode cache marker.

Each token mixer creates and consumes its own subclass (`KVCache` for full
attention, `ConvState` for linear attention); `Cache` lets the model hold a
``list[Cache]`` without naming both.

A batched cache is a pool: one buffer whose leading dimension is the batch,
carved into per-sequence slots. `Cache.view` narrows it to a contiguous row
range *as a view*, so a prefill can run batch-1 against one slot and a decode
can run batch-n against the first n, both writing the buffer the other reads.
`Cache.reset` clears a slot before a new sequence takes it.

Nothing narrows the *sequence* axis, because nothing needs to: attention takes
each row's length as a tensor rather than as a shape, so a step already reads
only as far as its sequence has got however long the slot is.
"""

from __future__ import annotations


class Cache:
    """Opaque per-layer decode state; only the owning mixer knows its shape."""

    def view(self, start: int, stop: int) -> Cache:
        """A view of rows ``[start, stop)`` sharing this cache's storage."""
        raise NotImplementedError

    def slot(self, index: int) -> Cache:
        """A single-row view, the shape a batch-1 prefill writes."""
        return self.view(index, index + 1)

    def reset(self, index: int) -> None:
        """Clear slot ``index``, so the next sequence starts from no context."""
        raise NotImplementedError

    def prime(self) -> None:
        """Mark this cache as holding state, whatever it currently holds.

        A pool is decoded against from the first step: its slots are zeroed
        rather than absent, and zeroed state is exactly "no context so far".
        Without this a mixer that branches on having state would take the
        prefill branch on a pool that is merely empty.
        """
