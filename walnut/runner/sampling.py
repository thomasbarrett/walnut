"""Drawing the next token, under whatever each row of the batch asked for.

The parameters are a request's (`walnut.scheduler.request.SamplingParams`);
only the draw is here, because the draw is the part that runs on the device.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from walnut.scheduler.request import SamplingParams


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

    def sample_batch(
        self,
        logits: torch.Tensor,
        params: Sequence[SamplingParams],
        generators: Sequence[torch.Generator | None] = (),
    ) -> torch.Tensor:
        """Draw one token per row of ``logits`` (B, vocab) under per-row params.

        Rows of a serving batch belong to different requests, so temperature,
        top-p and top-k vary down the batch. Rows that ask for the same thing
        are sampled together — one call covers the whole batch whenever the
        requests agree, which is the common case — and only genuinely different
        settings cost a second pass. A row with its own generator is its own
        group: `torch.multinomial` draws from one generator per call, so a
        seeded request cannot share a draw with anything else.
        """
        rows = len(params)
        gens: Sequence[torch.Generator | None] = generators or [None] * rows
        groups: dict[tuple, list[int]] = {}
        for row, param in enumerate(params):
            key = (
                (row,)
                if gens[row] is not None
                else (param.temperature, param.top_p, param.top_k)
            )
            groups.setdefault(key, []).append(row)

        if len(groups) == 1:
            return self(logits, params[0], gens[0])

        out = torch.empty(rows, 1, dtype=torch.long, device=logits.device)
        for members in groups.values():
            index = torch.tensor(members, device=logits.device)
            head = members[0]
            out[index] = self(logits[index], params[head], gens[head])
        return out
