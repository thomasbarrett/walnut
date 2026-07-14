"""Vision-tower building blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q, orig_k = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = ((q * cos) + (_rotate_half(q) * sin)).to(orig_q)
    k_embed = ((k * cos) + (_rotate_half(k) * sin)).to(orig_k)
    return q_embed, k_embed


class VisionPatchEmbed(nn.Module):
    """Patchify pixel values and project each patch to the vision hidden size."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        temporal_patch_size: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels, hidden_size, kernel_size=kernel, stride=kernel, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(target_dtype)).view(-1, self.hidden_size)


class VisionRotaryEmbedding(nn.Module):
    """2-D rotary position embedding for vision patches (height/width axes)."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class VisionAttention(nn.Module):
    """Bidirectional MHA over vision patches, attending within each image
    (``cu_seqlens`` boundaries) with 2-D rotary."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        # (seq, heads, dim) -> (1, heads, seq, dim)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # Block-diagonal mask: attend only within the same image.
        mask = torch.zeros(seq, seq, dtype=torch.bool, device=hidden_states.device)
        bounds = cu_seqlens.tolist()
        for i in range(1, len(bounds)):
            mask[bounds[i - 1] : bounds[i], bounds[i - 1] : bounds[i]] = True

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[None, None], scale=self.scaling
        )
        out = out.squeeze(0).transpose(0, 1).reshape(seq, -1)
        return self.proj(out)


class VisionPatchMerger(nn.Module):
    """Merge each ``spatial_merge_size**2`` patch group and project to
    ``out_hidden_size``. Patches arrive in merge-group order."""

    def __init__(
        self, hidden_size: int, out_hidden_size: int, spatial_merge_size: int
    ) -> None:
        super().__init__()
        self.merged_size = hidden_size * (spatial_merge_size**2)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_size, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))
