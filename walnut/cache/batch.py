"""How a pass reaches the cache: the addressing it resolves once, up front."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Batch:
    """Where each token of one forward pass reads and writes.

    Built by `CacheView.batch` once per pass and handed to every layer, so a
    change of layout is a change to one method rather than to every mixer.
    """

    #: (rows, width) — each token's ordinal in its own sequence. The cache's
    #: notion of position, which is not necessarily the rotary one.
    positions: torch.Tensor
    #: (rows,) int32 — each row's context length once this pass has written.
    #: What a varlen attention kernel reads to stop at a row's real context
    #: rather than at the width the pool allows.
    lengths: torch.Tensor
    #: (rows, width) — the cell each token writes, as a flat index into a
    #: token-addressed buffer's leading dimension. A slot pool makes these
    #: contiguous per row; a block pool will not, and nothing downstream cares.
    cells: torch.Tensor

    @property
    def rows(self) -> int:
        """Sequences in this pass."""
        return self.positions.shape[0]

    @property
    def width(self) -> int:
        """Tokens each row contributes to this pass: a prompt chunk, or one."""
        return self.positions.shape[1]
