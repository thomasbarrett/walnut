"""Where decode state lives: the pages a pass runs against, and who owns them."""

from __future__ import annotations

from bisect import insort
from collections.abc import Container

import torch

from walnut.cache.batch import Batch
from walnut.cache.state import Cache

#: Tokens per page. Not a tuning knob: the paged flash-attention kernel behind
#: `torch.nn.attention.varlen.varlen_attn` rejects any other value with "Paged
#: KV cache block size must be divisible by 256". It sets three things at once
#: — the granularity allocation rounds to, the internal fragmentation a
#: sequence carries (half a page on average), and, once a prefix tree sits
#: above `reserve`, the granularity at which two prompts can be found to agree.
PAGE_SIZE = 256


def pages_for(tokens: int) -> int:
    """Pages a sequence of ``tokens`` needs, rounding up to a whole page."""
    return -(-tokens // PAGE_SIZE)


class CacheView:
    """The per-layer state one pass runs against, and how it is laid out.

    A pass is handed a view rather than the pool because the rows it covers are
    part of the addressing: a prefill runs against one row and a decode against
    a bucket, and both resolve their cells through the block table of the rows
    they were given.
    """

    def __init__(self, caches: list[Cache], block_table: torch.Tensor) -> None:
        self.caches = caches
        #: (rows, pages) int32 — the physical page behind each logical page of
        #: each row. The whole of the placement rule now: sequence ``r``'s
        #: position ``p`` lives in cell ``block_table[r, p // PAGE_SIZE] *
        #: PAGE_SIZE + p % PAGE_SIZE``. A slot pool stated the same thing as a
        #: single stride, which is exactly what it could not outgrow.
        self.block_table = block_table
        self.rows = block_table.shape[0]
        self.max_k = block_table.shape[1] * PAGE_SIZE
        # Built once per view rather than per pass: a captured graph replays
        # the addresses it recorded, so anything a pass reads has to outlive
        # capture, and a view is built once where a pass runs every step.
        device = block_table.device
        self._cu = torch.arange(self.rows + 1, dtype=torch.int32, device=device)
        self._cu_k = self._cu * self.max_k

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
        """Rows ``[start, stop)`` of every layer, as one view.

        Only the block table is narrowed for a paged cache, and only the
        per-sequence caches narrow their storage: keys and values live in one
        pool of pages that belongs to no row in particular, so "rows
        ``[start, stop)``" is a statement about which sequences this pass
        addresses and not about which memory it may touch.
        """
        return CacheView(
            [cache.view(start, stop) for cache in self.caches],
            self.block_table[start:stop],
        )

    def row(self, index: int) -> CacheView:
        """A single-row view, what a batch-1 prefill runs against."""
        return self.view(index, index + 1)

    def batch(self, positions: torch.Tensor) -> Batch:
        """Resolve ``positions`` against this view's placement.

        ``positions`` is what the caller has: ``(width,)`` for a pass whose rows
        all sit at the same positions, which is a lone prefill; ``(rows, width)``
        for a batch of independent sequences; ``(3, rows, width)`` for M-RoPE,
        whose first axis is the temporal one and therefore the ordinal the cache
        wants.

        The page lookup is a `gather` rather than arithmetic on a stride, and
        it is done once here rather than per layer. It is also why this is
        safe to capture: the block table is a fixed buffer the scheduler
        writes before a replay, so a graph recorded against it reads whatever
        placement the current step has.
        """
        place = positions[0] if positions.ndim == 3 else positions
        if place.ndim == 1:
            place = place.expand(self.rows, -1)
        # A row attends through its own last position, so that position plus
        # one *is* its length.
        lengths = (place[:, -1] + 1).to(torch.int32)
        pages = self.block_table.gather(1, place // PAGE_SIZE)
        cells = pages * PAGE_SIZE + place % PAGE_SIZE
        width = place.shape[1]
        # Decode is width 1, where the packed query offsets are the row
        # ordinals themselves; only a prefill pays the multiply.
        cu_q = self._cu if width == 1 else self._cu * width
        return Batch(
            place, lengths, cells, self.block_table, cu_q, self._cu_k, self.max_k
        )


class CachePool(CacheView):
    """Every layer's decode state, and which rows and pages are free.

    A pool is the view of every row, plus the right to hand placements out.
    There are two resources and they are not the same shape. A *row* is a
    sequence's identity — the slot its recurrent state occupies, and the line
    of the block table it addresses through — and rows are bounded by
    ``max_batch_size`` because a decode step runs them all. A *page* is 256
    tokens of key/value storage, drawn from one pool shared by every row, and
    a sequence takes only as many as its prompt and completion need rather
    than as many as ``max_seq_len`` allows.

    That is the whole of what paging buys here: under a slot pool a request
    asking for 200 tokens reserved the same memory as one asking for 8192, so
    the batch was sized for the worst case every request might be. Both are
    still preallocated and neither is overcommitted — a request that does not
    fit waits rather than displacing a running one — but the second resource
    is now sized to demand.

    One page is held back as *scratch*, and every row that holds no sequence
    points its whole block table at it. A decode step runs the rows of its
    bucket whether or not anyone holds them, and each of those rows writes a
    key and a value somewhere; under a slot pool "somewhere" was the row's own
    unused slot and cost nothing, but a paged row with an empty block table
    addresses page 0, so every idle row in the bucket would write over the
    first token of whichever sequence happened to hold it. The scratch page is
    where those writes go instead.

    `reserve` and `release` are still the only way in, which is where a prefix
    tree attaches: it hands back placements whose pages overlap.
    """

    #: The page idle rows write into. Zero, so a zeroed block table already
    #: points at it and a released row needs only to be cleared.
    SCRATCH = 0

    def __init__(
        self,
        caches: list[Cache],
        max_batch_size: int,
        max_seq_len: int,
        pages: int | None = None,
    ) -> None:
        per_row = pages_for(max_seq_len)
        device = (
            caches[0].buffers()[0].device if caches and caches[0].buffers() else None
        )
        super().__init__(
            caches,
            torch.zeros(max_batch_size, per_row, dtype=torch.int32, device=device),
        )
        self.max_seq_len = max_seq_len
        #: Pages sequences can hold. At parity with a slot pool by default —
        #: every row could still run to ``max_seq_len`` at once — so raising
        #: ``max_batch_size`` is what spends the slack, not the default. The
        #: storage behind it is one page larger, for `SCRATCH`.
        self.pages = max_batch_size * per_row if pages is None else pages
        self._free_rows = list(range(max_batch_size))
        self._free_pages = list(range(self.SCRATCH + 1, self.pages + 1))
        self._held: dict[int, list[int]] = {}

    def reserve(self, tokens: int, shared: int = 0) -> int | None:
        """Take a row and the pages for ``tokens``, or None if either is short.

        Lowest free row first, which keeps the live set dense — a decode step
        runs whole buckets, so a sequence parked in a high row costs every step
        the rows beneath it. Pages come off the free list in whatever order
        they were returned, because a paged read does not care.

        ``shared`` says the first that many pages are coming from somewhere
        else — a prefix tree lending pages another sequence already built — so
        they are neither allocated nor written into the block table here. The
        caller fills them in; see `walnut.cache.radix.PrefixCache`.
        """
        # At least one page even for a zero-token reservation: a row with an
        # empty block table addresses `SCRATCH`, which is not storage anyone
        # may keep a sequence in.
        want = max(1, pages_for(tokens))
        need = want - shared
        if not self._free_rows or need > len(self._free_pages) or want > self.per_row:
            return None
        # A sequence always writes somewhere: `shared` counts pages of a prompt
        # it can skip, and it was never allowed to skip the whole thing.
        assert need > 0
        row = self._free_rows.pop(0)
        pages = [self._free_pages.pop(0) for _ in range(need)]
        self._held[row] = pages
        table = self.block_table[row]
        # Cleared first, so the pages past this sequence's own point at
        # `SCRATCH` rather than at what the last holder of this row was using.
        table.zero_()
        table[shared:want] = torch.tensor(pages, dtype=torch.int32, device=table.device)
        return row

    def held(self, row: int) -> tuple[int, ...]:
        """The pages ``row`` was allocated, in the order it writes them."""
        return tuple(self._held.get(row, ()))

    def free_page(self, page: int) -> None:
        """Return one page held outside any row, as a prefix tree holds them."""
        insort(self._free_pages, page)

    def release(self, row: int, keep: Container[int] = ()) -> None:
        """Give a finished sequence's row and pages back.

        ``keep`` names pages that are not going back to the free list because
        something outlasting the sequence has taken them over — a prefix tree
        grafting the cache this sequence built onto a shared trie. They are
        still this row's to give up; they are just not free afterwards.

        Clearing the block table is not tidying: it points the row at `SCRATCH`
        again, which is what keeps the decode steps it goes on taking as an
        idle row from landing in somebody else's pages.
        """
        pages = self._held.pop(row, None)
        if pages is None:
            return
        self.block_table[row].zero_()
        for page in pages:
            if page not in keep:
                insort(self._free_pages, page)
        self._free_rows.append(row)
        self._free_rows.sort()

    @property
    def per_row(self) -> int:
        """Pages one row's block table can address: ``max_seq_len`` rounded up."""
        return self.block_table.shape[1]

    @property
    def free_rows(self) -> int:
        """Rows a new sequence could take."""
        return len(self._free_rows)

    @property
    def free_pages(self) -> int:
        """Pages a new sequence could take."""
        return len(self._free_pages)

    def fits(self, tokens: int) -> bool:
        """Whether a sequence of ``tokens`` could *ever* fit, pool empty."""
        return pages_for(tokens) <= min(self.pages, self.per_row)

    def reset(self, row: int) -> None:
        """Clear ``row`` in every layer, before a new sequence takes it.

        Which is now only the per-sequence caches: a page handed to a new
        sequence still holds the last one's keys, and never has to be cleared,
        because a row's length bounds what attention reads and every cell
        inside that bound was written by this sequence.
        """
        for cache in self.caches:
            cache.reset(row)

    def prime(self) -> None:
        """Tell every layer its buffers count as state.

        The pool is decoded against from the first step: its rows hold zeros,
        which is what "no context yet" means, and without this a mixer that
        branches on having state would take the prefill branch on a pool that
        is merely empty.
        """
        for cache in self.caches:
            cache.prime()

    def carried(self, row: int) -> list[torch.Tensor]:
        """``row``'s state that a decode step advances rather than indexes."""
        return [tensor for cache in self.caches for tensor in cache.carried(row)]
