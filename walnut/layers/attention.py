"""Grouped-query softmax attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn.attention.varlen import varlen_attn

from walnut.cache import PAGE_SIZE, Batch, TokenCache


class KVCache(TokenCache):
    """Static, pre-allocated key/value pages: one pool, shared by every row.

    Laid out ``(2, pages, PAGE_SIZE, head, dim)`` — keys and values as the two
    halves of one buffer, each contiguous, which is the shape the paged kernel
    reads and the shape one allocation can hold. There is no batch axis: a
    page belongs to whichever sequence is currently holding it, and `Batch`
    is what says which those are.

    Writing still goes through `Batch.cells`, a flat index into the pool's
    ``pages * PAGE_SIZE`` cells. Reading cannot: a row's context is scattered
    across pages, so the kernel is handed the row's block table and walks it.
    """

    def __init__(
        self,
        pages: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        self.kv = torch.zeros(
            2, pages, PAGE_SIZE, n_kv_heads, head_dim, dtype=dtype, device=device
        )

    @property
    def k(self) -> torch.Tensor:
        """The key pages, ``(pages, PAGE_SIZE, head, dim)``."""
        return self.kv[0]

    @property
    def v(self) -> torch.Tensor:
        """The value pages, ``(pages, PAGE_SIZE, head, dim)``."""
        return self.kv[1]

    def buffers(self) -> list[torch.Tensor]:
        return [self.kv]

    def write(self, batch: Batch, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write ``k``/``v`` (rows, width, H, D) into the cells ``batch`` names.

        One flat scatter over both halves at once, whatever the pass is: a
        prompt chunk writing a run of cells and a decode step writing one cell
        per row differ only in the indices they are handed. Which is the point
        — a placement this does not have to understand is a placement that can
        change.

        Keys and values are one tensor precisely so this is one scatter.
        Indexing a separate ``k`` and ``v`` by cell was two, and cost a decode
        step one extra kernel per full-attention layer over the ``(row,
        position)`` indexing a slot pool allowed; stacking the two writes into
        the leading axis of one buffer takes that back.
        """
        heads, dim = self.kv.shape[3], self.kv.shape[4]
        cells = batch.cells.reshape(-1)
        self.kv.view(2, -1, heads, dim)[:, cells] = torch.stack(
            (k.reshape(-1, heads, dim), v.reshape(-1, heads, dim))
        )


@dataclass(frozen=True)
class KVCacheSpec:
    """`KVCache`, as a size and a way to build it.

    Sized by pages and not by rows: keys and values live in one pool that
    belongs to no row in particular, which is the whole of what paging bought.
    """

    heads: int
    dim: int

    def nbytes(self, rows: int, pages: int, dtype: torch.dtype) -> int:
        # (2, pages, PAGE_SIZE, heads, dim) — keys and values, as one buffer.
        return 2 * pages * PAGE_SIZE * self.heads * self.dim * dtype.itemsize

    def build(
        self,
        rows: int,
        pages: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> KVCache:
        return KVCache(pages, self.heads, self.dim, dtype, device)


@torch._dynamo.disable
def _varlen(
    q: torch.Tensor, cache: KVCache, batch: Batch, scale: float
) -> torch.Tensor:
    """``q`` (B, S, H, D) against each row's own pages, as `batch` maps them.

    The rectangle is what makes a cached step expensive: a row's context is as
    long as that row has got, but one `scaled_dot_product_attention` has to
    span the longest slot and mask the rest — which charges every request for
    the context length the pool *allows* rather than the one it is using, and
    materializes a mask and a GQA-expanded copy of the whole buffer to do it.
    `varlen_attn` takes the per-row length as a tensor instead, so the kernel
    reads each row's real context, and length being data rather than shape is
    what lets one captured graph serve a row at any point in its sequence.

    ``block_table`` is the paged half of the same idea: the row's keys are not
    a contiguous run any more, so the kernel is told which page holds each of
    its logical pages and follows that. ``cu_seq_k`` goes unread on this path
    — the extent comes from ``seqused_k`` and the table — but the operator
    requires it to be set alongside ``cu_seq_q``.

    It covers both phases. Causal here aligns each row's last query with its
    last key, so ``S`` queries against ``S`` keys is a prompt's causal mask and
    one query against ``n`` keys is a decode step attending to all of them.

    ``num_splits=1`` is a correctness workaround, not a tuning choice. Splitting
    the key axis across blocks and combining the partial softmaxes is how flash
    attention finds parallelism when the batch is too small to fill the GPU,
    and on this build the combine is wrong for a paged batch of more than one
    row: rows past the first come back with errors of order 1, and often NaN or
    1e38, at every split count above 1 and under every ``cu_seq_k`` convention.
    One split is exact. It costs nothing today, because the contiguous path
    this replaces was not splitting either, but it is holding a real number
    down — the same shapes with splits enabled run 2-6x faster — so this line
    is where that comes back if the kernel is fixed.

    Kept out of the compiled region because dynamo does not preserve the call:
    traced, it decomposes back into a generic attention, which is 8% of TPOT
    and produces the token stream the kernel this replaced produced. The graph
    break costs one launch per full-attention layer, and `walnut.runner.graphs`
    captures across it anyway.
    """
    rows, seq, heads, head_dim = q.shape
    out = varlen_attn(
        q.reshape(rows * seq, heads, head_dim),
        cache.k,
        cache.v,
        batch.cu_q,
        batch.cu_k,
        seq,
        batch.max_k,
        scale=scale,
        enable_gqa=True,
        window_size=(-1, 0),
        seqused_k=batch.lengths,
        block_table=batch.block_table,
        num_splits=1,
    )
    return cast(torch.Tensor, out).view(rows, seq, heads, head_dim)


class Attention(nn.Module):
    """Causal GQA attention over (B, S, heads, head_dim) q/k/v, through a KV cache.

    The cache is not optional. Every attention this model does is part of a
    sequence being generated, so it is always writing its keys and values into
    the pages it holds and reading them back; there is no cacheless call to
    serve.

    Where those cells are, and how long each row's context runs, arrive in a
    `Batch` the pass computed once. This layer holds the head geometry and the
    kernel call, and nothing about the layout.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

    def cache_spec(self) -> KVCacheSpec:
        """What this layer's keys and values cost, before they are allocated."""
        return KVCacheSpec(self.num_kv_heads, self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: KVCache,
        batch: Batch,
    ) -> torch.Tensor:
        cache.write(batch, k, v)
        return _varlen(q, cache, batch, self.scaling)
