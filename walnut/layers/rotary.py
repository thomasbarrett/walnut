"""Rotary position embeddings (RoPE) with partial rotary and interleaved M-RoPE."""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim to the front (negated)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the leading rotary_dim dims of q/k (B, S, heads, head_dim), passing
    the tail through (partial rotary)."""
    cos = cos.unsqueeze(-2).to(q.dtype)
    sin = sin.unsqueeze(-2).to(q.dtype)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """M-RoPE cos/sin for (3, batch, seq) position ids; rotates
    head_dim * partial_rotary_factor dims and interleaves the t/h/w axes."""

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        partial_rotary_factor: float = 1.0,
        mrope_section: list[int] | None = None,
    ) -> None:
        super().__init__()
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = mrope_section

    def _interleave_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Interleave the 3 axes: [TTT..HHH..WWW] -> [THWTHW..]."""
        if self.mrope_section is None:
            return freqs[0]
        freqs_t = freqs[0]
        for axis, offset in enumerate((1, 2), start=1):  # height, width
            length = self.mrope_section[axis] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[axis, ..., idx]
        return freqs_t

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            positions = positions[None, None, :].expand(3, 1, -1)
        elif positions.ndim == 2:
            positions = positions[None].expand(3, positions.shape[0], -1)

        inv_freq = self.inv_freq[None, None, :, None].float()
        inv_freq = inv_freq.expand(3, positions.shape[1], -1, 1).to(positions.device)
        pos = positions[:, :, None, :].float()
        freqs = (inv_freq @ pos).transpose(2, 3)  # (3, bs, seq, dim//2)
        freqs = self._interleave_mrope(freqs)  # (bs, seq, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (bs, seq, dim)
        return emb.cos(), emb.sin()
