"""What a layer remembers, and which kinds of it can be paged and shared."""

from __future__ import annotations

import torch


class Cache:
    """One layer's decode state, across every sequence the pool holds.

    Subclasses declare their storage through `buffers` and narrow it through
    `view`; resetting and saving a slot follow from those, so a new kind of
    state is two methods rather than five.
    """

    def buffers(self) -> list[torch.Tensor]:
        """Every mutable tensor this cache owns, in a stable order.

        The one place that enumerates state. CUDA graph capture writes the
        cache for real and has to put it back, and a slot handed to a new
        sequence has to be cleared; both work off this rather than off knowing
        what kind of cache they hold.
        """
        raise NotImplementedError

    def view(self, start: int, stop: int) -> Cache:
        """A view of rows ``[start, stop)`` sharing this cache's storage.

        Sharing, not copying: a prefill running batch-1 against one slot and a
        decode running batch-n against the first n write the same buffers.
        """
        raise NotImplementedError

    def slot(self, index: int) -> Cache:
        """A single-row view, the shape a batch-1 prefill writes."""
        return self.view(index, index + 1)

    def reset(self, index: int) -> None:
        """Clear slot ``index``, so the next sequence starts from no context.

        Zeroed is what "no context yet" means to every cache here: an unwritten
        key is never read, because a row's length bounds what attention sees,
        and zeroed recurrent state is the identity the recurrence starts from.
        """
        for buffer in self.buffers():
            buffer[index].zero_()

    def carried(self, index: int) -> list[torch.Tensor]:
        """Slot ``index``'s tensors that a step advances rather than indexes.

        A decode step runs every row of its batch, live or not, so a slot part
        way through a chunked prefill is stepped along with the rest. A
        positional write survives that — the scheduler aims the row at the
        position its next chunk overwrites — but recurrent state has nowhere to
        be aimed: a step moves it on, and the prompt's context is gone. These
        are the tensors `walnut.scheduler` saves across a step and puts back.
        """
        return []

    def prime(self) -> None:
        """Mark this cache as holding state, whatever it currently holds."""


class TokenCache(Cache):
    """State stored per token: one addressable cell per position.

    Grows with the sequence, and every cell belongs to exactly one token — which
    is what makes this kind splittable into blocks, and shareable between two
    sequences whose prompts agree. `Batch.cells` is the only thing that says
    which cell is which, so paging one is a change of placement, not of layout.
    """


class StateCache(Cache):
    """State stored per sequence: a fixed summary of everything before it.

    Neither pageable nor shareable — there is no cell to hand to a second
    sequence, only a running state that a step advances. Which is exactly what
    `carried` is for, and why it is defined here and not per subclass.
    """

    #: Whether a forward pass has written state into these buffers. A pool is
    #: decoded against from its first step, so its slots hold zeros rather than
    #: nothing, and a mixer branching on "has state" needs to be told the
    #: difference between an empty pool and an empty sequence.
    primed: bool = False

    def prime(self) -> None:
        self.primed = True

    @property
    def empty(self) -> bool:
        """True until a forward pass has written state into the buffers."""
        return not self.primed

    def carried(self, index: int) -> list[torch.Tensor]:
        """Every buffer's slot ``index``: all of it is carried, by definition."""
        return [buffer[index] for buffer in self.buffers()]
