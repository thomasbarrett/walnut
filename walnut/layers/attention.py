"""Grouped-query softmax attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from walnut.layers.cache import Cache


class KVCache(Cache):
    """Static, pre-allocated key/value cache: fixed-size buffers written by
    slot, so tensor shapes stay identical across prefill and decode steps."""

    def __init__(
        self,
        max_batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        shape = (max_batch_size, n_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``k``/``v`` (B, H, S, D) into slots ``input_pos`` (S,); return
        the full buffers (B, H, max_seq, D)."""
        self.k[:, :, input_pos] = k
        self.v[:, :, input_pos] = v
        return self.k, self.v


class Attention(nn.Module):
    """Causal GQA attention over (B, S, heads, head_dim) q/k/v; optional KV cache."""

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
        cache: KVCache | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # With a cache, k/v span the whole buffer, so a mask (not is_causal)
        # limits each query to slots up to its own position.
        mask = None
        causal = True
        if cache is not None:
            assert input_pos is not None
            k, v = cache.update(input_pos, k, v)
            key_pos = torch.arange(k.shape[2], device=k.device)
            mask = (key_pos[None, :] <= input_pos[:, None])[None, None]
            causal = False

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal, scale=self.scaling
        )
        return out.transpose(1, 2).contiguous()
