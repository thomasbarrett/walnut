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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * (1.0 + self.weight.float())).to(input_dtype)
