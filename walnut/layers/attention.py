"""Grouped-query softmax attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class KVCache:
    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None


class Attention(nn.Module):
    """Causal GQA attention over (B, S, heads, head_dim) q/k/v; optional KV cache."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cache is not None:
            past_k, past_v = cache.k, cache.v
            if past_k is not None and past_v is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            cache.k = k
            cache.v = v

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Causal only when q and k lengths match (prefill); a decode step's single
        # query attends every cached key.
        causal = q.shape[2] == k.shape[2]
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=self.scaling
        )
        return out.transpose(1, 2).contiguous()
