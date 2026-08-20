"""Grouped-query softmax attention."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn.attention.varlen import varlen_attn

from walnut.cache import Batch, TokenCache


class KVCache(TokenCache):
    """Static, pre-allocated key/value cache: fixed-size buffers written by
    cell, so tensor shapes stay identical across prefill and decode steps.

    Laid out ``(batch, position, head, dim)`` and written through a flat index
    into the first two axes together, which is what `Batch.cells` names. A slot
    pool makes a sequence's cells one contiguous run, which is the layout
    `varlen_attn` reads and the layout q/k/v already arrive in; a block pool
    would scatter them, and only the read side would have to change.
    """

    def __init__(
        self,
        max_batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        shape = (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self._index()

    def _index(self) -> None:
        """The slot boundaries `varlen_attn` reads.

        Built once per view rather than per step: a captured graph replays
        whatever addresses it recorded, so anything it reads has to outlive the
        capture.
        """
        rows, slot = self.k.shape[0], self.k.shape[1]
        self.cu_seqlens = torch.arange(
            0, (rows + 1) * slot, slot, dtype=torch.int32, device=self.k.device
        )

    @classmethod
    def _view(cls, k: torch.Tensor, v: torch.Tensor) -> KVCache:
        cache = cls.__new__(cls)
        cache.k, cache.v = k, v
        cache._index()
        return cache

    def buffers(self) -> list[torch.Tensor]:
        return [self.k, self.v]

    def view(self, start: int, stop: int) -> KVCache:
        """Rows ``[start, stop)``, spanning whole slots.

        Whole slots because `_varlen` reads one by stride: narrowing the
        sequence axis would cost it its layout to save work it does not do.
        """
        return KVCache._view(self.k[start:stop], self.v[start:stop])

    def write(self, batch: Batch, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write ``k``/``v`` (rows, width, H, D) into the cells ``batch`` names.

        One flat scatter, whatever the pass is: a prompt chunk writing a run of
        cells and a decode step writing one cell per row differ only in the
        indices they are handed. Which is the point — a placement this does not
        have to understand is a placement that can change.

        It costs something today. Indexing by ``(row, position)``, as this did
        while a slot was the only placement there was, let Inductor fuse the
        two writes into one kernel; indexing by cell does not, so a decode step
        carries one extra kernel per full-attention layer. Same total GPU time
        — the kernels do the same work — but a captured graph has more nodes to
        walk, which measures as 0.7% of TPOT on a 0.8B model. Paging takes it
        back: a block store holds keys and values in one tensor, and one
        scatter writes both.
        """
        heads, dim = self.k.shape[2], self.k.shape[3]
        cells = batch.cells.reshape(-1)
        self.k.view(-1, heads, dim)[cells] = k.reshape(-1, heads, dim)
        self.v.view(-1, heads, dim)[cells] = v.reshape(-1, heads, dim)


@torch._dynamo.disable
def _varlen(
    q: torch.Tensor,
    cache: KVCache,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """``q`` (B, S, H, D) against ``lengths`` of each row's own cache slot.

    The rectangle is what makes a cached step expensive: a row's context is as
    long as that row has got, but one `scaled_dot_product_attention` has to
    span the longest slot and mask the rest — which charges every request for
    the context length the pool *allows* rather than the one it is using, and
    materializes a mask and a GQA-expanded copy of the whole buffer to do it.
    `varlen_attn` takes the per-row length as a tensor instead, so the kernel
    reads each row's real context, and length being data rather than shape is
    what lets one captured graph serve a slot at any point in its sequence.

    It covers both phases. Causal here aligns each row's last query with its
    last key, so ``S`` queries against ``S`` keys is a prompt's causal mask and
    one query against ``n`` keys is a decode step attending to all of them.

    Kept out of the compiled region because dynamo does not preserve the call:
    traced, it decomposes back into a generic attention, which is 8% of TPOT
    and produces the token stream the kernel this replaced produced. The graph
    break costs one launch per full-attention layer, and `walnut.graph`
    captures across it anyway.
    """
    batch, seq, heads, head_dim = q.shape
    kv_heads = cache.k.shape[2]
    slot = cache.k.shape[1]
    out = varlen_attn(
        q.reshape(batch * seq, heads, head_dim),
        cache.k.view(batch * slot, kv_heads, head_dim),
        cache.v.view(batch * slot, kv_heads, head_dim),
        cache.cu_seqlens[: batch + 1] // slot * seq,
        cache.cu_seqlens[: batch + 1],
        seq,
        slot,
        scale=scale,
        enable_gqa=True,
        window_size=(-1, 0),
        seqused_k=lengths,
    )
    return cast(torch.Tensor, out).view(batch, seq, heads, head_dim)


class Attention(nn.Module):
    """Causal GQA attention over (B, S, heads, head_dim) q/k/v, through a KV cache.

    The cache is not optional. Every attention this model does is part of a
    sequence being generated, so it is always writing its keys and values into
    a slot and reading that slot back; there is no cacheless call to serve.

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

    def make_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> KVCache:
        return KVCache(
            max_batch_size, self.num_kv_heads, max_seq_len, self.head_dim, dtype, device
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: KVCache,
        batch: Batch,
    ) -> torch.Tensor:
        cache.write(batch, k, v)
        return _varlen(q, cache, batch.lengths, self.scaling)
