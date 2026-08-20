"""How a pass reaches the cache: the addressing it resolves once, up front."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Batch:
    """Where each token of one forward pass reads and writes.

    Built by `CacheView.batch` once per pass and handed to every layer, so a
    change of layout is a change to one method rather than to every mixer.

    Reading and writing are addressed differently, which is what paging costs
    and why both are here. A write goes to one cell, so `cells` names it
    directly; a read walks a whole row's context, so the kernel is given the
    row's `block_table` and follows it page by page. A slot pool could state
    both as one stride.
    """

    #: (rows, width) — each token's ordinal in its own sequence. The cache's
    #: notion of position, which is not necessarily the rotary one.
    positions: torch.Tensor
    #: (rows,) int32 — each row's context length once this pass has written.
    #: What a varlen attention kernel reads to stop at a row's real context
    #: rather than at the width the pool allows.
    lengths: torch.Tensor
    #: (rows, width) — the cell each token writes, as a flat index into a
    #: token-addressed buffer's leading dimension. Contiguous per row under a
    #: slot pool; scattered across pages under this one, and nothing that
    #: writes through it can tell the difference.
    cells: torch.Tensor
    #: (rows, pages) int32 — the physical page holding each logical page of a
    #: row's context, which is how a paged attention kernel reads a row whose
    #: cells are not contiguous. Entries past a row's length are never read.
    block_table: torch.Tensor
    #: (rows + 1,) int32 — where each row's queries start in this pass, packed.
    #: The kernel takes its batch as one ragged run rather than a rectangle.
    cu_q: torch.Tensor
    #: (rows + 1,) int32 — the same for keys, in the units a full row spans.
    #: The paged kernel derives each row's real extent from `lengths` and
    #: `block_table` and does not consult this, but it is required to be set
    #: and this is what it would mean.
    cu_k: torch.Tensor
    #: Tokens a row's block table can address, ``pages * page_size``. The
    #: kernel's bound on how far a row could reach, not how far it does.
    max_k: int

    @property
    def rows(self) -> int:
        """Sequences in this pass."""
        return self.positions.shape[0]

    @property
    def width(self) -> int:
        """Tokens each row contributes to this pass: a prompt chunk, or one."""
        return self.positions.shape[1]
