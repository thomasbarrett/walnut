"""Projections that share an input, held as one matmul."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class FusedLinear(nn.Module):
    """Several of a checkpoint's projections, held as one linear layer.

    Projections reading the same input compute the same values as one matmul
    over the concatenated weight, split after. At decode's batch of 1 each is a
    bandwidth-bound gemv, so the group costs one kernel instead of several — and
    the narrow ones stop paying the launch floor for the few values they make.

    ``parts`` maps the name of each projection the checkpoint stores separately
    to the width it contributes, in concatenation order. It is the only place
    the fusion is written down: `forward` splits by these widths and
    `walnut.models.loader.copy_weights` fills the weight from these names, so
    the split and the load cannot drift apart.

    Parameters are named ``weight`` and ``bias``, as `torch.nn.Linear` names
    them, so a fused layer is otherwise an ordinary linear layer to the rest of
    the engine.

    Parts concatenate along dim 0, the way the checkpoint's separate
    projections stack. Were tensor parallelism to land, note that grouped-query
    attention wants a head-grouped layout instead — ``[kv_heads, heads_per_kv +
    2, head_dim]``, as torchtitan and MaxText use — for which a plain dim-0
    shard is correct; a flat concatenation would split a rank's queries from
    its keys.
    """

    def __init__(
        self, in_features: int, parts: dict[str, int], bias: bool = False
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = sum(parts.values())
        self.parts = parts
        self.weight = nn.Parameter(torch.empty(self.out_features, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise as `torch.nn.Linear` does, so the two are interchangeable."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One matmul, then one tensor per part along the last dimension.

        The parts are strided views, not contiguous: reshaping one needs
        ``reshape`` rather than ``view`` unless the input is a single token.
        """
        out = F.linear(x, self.weight, self.bias)
        return out.split(list(self.parts.values()), dim=-1)

    def extra_repr(self) -> str:
        parts = ", ".join(f"{name}={width}" for name, width in self.parts.items())
        return f"in_features={self.in_features}, {parts}, bias={self.bias is not None}"
