"""What a layer remembers, and which kinds of it can be paged and shared."""

from __future__ import annotations

import torch


class Cache:
    """One layer's decode state, across every sequence the pool holds.

    Subclasses declare their storage through `buffers` and narrow it through
    `view`; resetting and saving a row follow from those, so a new kind of
    state is two methods rather than five.
    """

    def buffers(self) -> list[torch.Tensor]:
        """Every mutable tensor this cache owns, in a stable order.

        The one place that enumerates state. CUDA graph capture writes the
        cache for real and has to put it back, and a row handed to a new
        sequence has to be cleared; both work off this rather than off knowing
        what kind of cache they hold.
        """
        raise NotImplementedError

    def view(self, start: int, stop: int) -> Cache:
        """This cache as rows ``[start, stop)`` address it.

        Sharing, not copying: a prefill running batch-1 against one row and a
        decode running batch-n against the first n write the same buffers.
        What narrowing *means* depends on the kind — see `StateCache`, which
        holds a row per sequence, and `TokenCache`, which holds none.
        """
        raise NotImplementedError

    def row(self, index: int) -> Cache:
        """A single-row view, the shape a batch-1 prefill writes."""
        return self.view(index, index + 1)

    def reset(self, index: int) -> None:
        """Clear row ``index``, so the next sequence starts from no context.

        Zeroed is what "no context yet" means to every cache here: an unwritten
        key is never read, because a row's length bounds what attention sees,
        and zeroed recurrent state is the identity the recurrence starts from.
        """
        for buffer in self.buffers():
            buffer[index].zero_()

    def carried(self, index: int) -> list[torch.Tensor]:
        """Row ``index``'s tensors that a step advances rather than indexes.

        A decode step runs every row of its batch, live or not, so a row part
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

    Grows with the sequence, and every cell belongs to exactly one token —
    which is what makes this kind splittable into pages, and shareable between
    two sequences whose prompts agree. `Batch.cells` is the only thing that
    says which cell is which, so paging one is a change of placement, not of
    layout.

    Paged, its storage stops belonging to rows at all: one pool of pages, and
    a block table saying which of them a sequence is currently holding. Both
    methods here follow from that, and both of them stop doing anything.
    """

    def view(self, start: int, stop: int) -> Cache:
        """Itself. A page pool has no row axis to narrow.

        Which rows a pass addresses is `Batch`'s answer, through the block
        table it was built from; the storage a pass may touch is the whole
        pool either way, because a row's pages are wherever they were free.
        """
        return self

    def reset(self, index: int) -> None:
        """Nothing. A cell is always written before it is read.

        A page handed to a new sequence still holds the last one's keys.
        Attention stops at the row's own length, and every cell inside that
        length was written by this sequence on its way there, so the stale
        remainder is unreachable rather than merely wrong. Zeroing it was the
        single largest cost of admitting a request under a slot pool — a whole
        row of every layer, on the time-to-first-token path.
        """


class StateCache(Cache):
    """State stored per sequence: a fixed summary of everything before it.

    Not pageable — there is no cell to split off, only a running state that a
    step advances — which is why a row remains a resource in its own right
    even once keys and values stop needing one. Sharing it between two
    sequences with a common prompt is possible but is not free the way sharing
    a page is: the state after n tokens is the same for both, and handing the
    second one a pointer would also hand it the right to advance it, so a
    prefix tree has to *copy* this where it merely references a page.

    `carried` is what a step advancing it costs elsewhere, and is defined here
    rather than per subclass for the same reason.
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
        """Every buffer's row ``index``: all of it is carried, by definition."""
        return [buffer[index] for buffer in self.buffers()]
