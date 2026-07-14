"""Token sampling for generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class SamplingParams:
    max_new_tokens: int = 20
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)
    seed: int | None = None


class Sampler(nn.Module):
    """Draw a next token per row of logits (B, vocab) -> (B, 1)."""

    def forward(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if params.temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / params.temperature

        if params.top_k > 0:
            k = min(params.top_k, logits.shape[-1])
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        probs = torch.softmax(logits, dim=-1)

        if params.top_p < 1.0:
            order = probs.sort(descending=True, dim=-1)
            # Drop tokens past the point where the running mass first exceeds top_p.
            mask = order.values.cumsum(dim=-1) - order.values > params.top_p
            kept = order.values.masked_fill(mask, 0.0)
            probs = torch.zeros_like(probs).scatter(-1, order.indices, kept)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1, generator=generator)
