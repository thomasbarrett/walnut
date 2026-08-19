"""Grouped-query softmax attention."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn.attention.varlen import varlen_attn

from walnut.layers.cache import Cache


class KVCache(Cache):
    """Static, pre-allocated key/value cache: fixed-size buffers written by
    slot, so tensor shapes stay identical across prefill and decode steps.

    The batch dimension is a pool of sequence slots (see `Cache`). Writes are
    positional per row, so a batch whose rows sit at different points in their
    own sequences still lands in one indexed store.

    Laid out ``(batch, position, head, dim)``: each slot is then a contiguous
    run of tokens, which is the layout `varlen_attn` reads a cache in, and the
    layout q/k/v already arrive in.
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
        """Row numbers, and the slot boundaries `varlen_attn` reads.

        Both are built once per view rather than per step: a captured graph
        replays whatever addresses it recorded, so anything it reads has to
        outlive the capture.
        """
        rows, slot = self.k.shape[0], self.k.shape[1]
        device = self.k.device
        self.rows_index = torch.arange(rows, device=device)
        self.cu_seqlens = torch.arange(
            0, (rows + 1) * slot, slot, dtype=torch.int32, device=device
        )

    @classmethod
    def _view(cls, k: torch.Tensor, v: torch.Tensor) -> KVCache:
        cache = cls.__new__(cls)
        cache.k, cache.v = k, v
        cache._index()
        return cache

    def view(self, start: int, stop: int) -> KVCache:
        """Rows ``[start, stop)``, spanning whole slots.

        Whole slots because `_varlen` reads one by stride: narrowing the
        sequence axis would cost it its layout to save work it does not do.
        """
        return KVCache._view(self.k[start:stop], self.v[start:stop])

    def reset(self, index: int) -> None:
        self.k[index].zero_()
        self.v[index].zero_()

    def update(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``k``/``v`` (B, S, H, D) into the slots ``input_pos`` names;
        return the full buffers (B, max_seq, H, D).

        ``input_pos`` is either (S,) — every row of the batch at the same
        positions, which is what a lone prefill produces — or (B, S), one
        position per row, which is what a batch of independent sequences needs.
        """
        if input_pos.ndim == 1:
            self.k[:, input_pos] = k
            self.v[:, input_pos] = v
        else:
            rows = self.rows_index[:, None]
            self.k[rows, input_pos] = k
            self.v[rows, input_pos] = v
        return self.k, self.v


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
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        cache.update(input_pos, k, v)
        # A row attends through its own last position, so that position plus
        # one *is* its length.
        positions = (
            input_pos.reshape(q.shape[0], -1)
            if input_pos.ndim > 1
            else input_pos.expand(q.shape[0], -1)
        )
        lengths = (positions[:, -1] + 1).to(torch.int32)
        return _varlen(q, cache, lengths, self.scaling)
