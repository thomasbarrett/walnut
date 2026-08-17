"""RMS normalization."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS norm with Qwen's 1-centered scale: norm(x) * (1 + weight)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        # `1 + weight` in float32, built on first use. Not persistent: it is
        # derived from `weight`, so it belongs in neither the checkpoint nor
        # the state dict, but registering it as a buffer still has it follow
        # `.to()` onto the model's device and survive `load_state_dict`.
        self.register_buffer("_scale", None, persistent=False)
        # A reload replaces the weight the cache was derived from.
        self.register_load_state_dict_post_hook(lambda self, _: self._invalidate())

    def _invalidate(self) -> None:
        self._scale = None

    @property
    def scale(self) -> torch.Tensor:
        """`1 + weight` in float32, cached across calls.

        A constant of the loaded checkpoint, but computing it inline costs a
        `.float()` and an `add` on every call — two of the ~7 dispatches this
        module spends, in the module prefill runs most (79 times on a 19-token
        prompt, 22% of its kernels). It cannot be built in ``__init__``,
        because the weight arrives afterwards.
        """
        if self._scale is None:
            self._scale = 1.0 + self.weight.float()
        return self._scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.scale).to(input_dtype)
