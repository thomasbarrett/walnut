"""Gated-DeltaNet linear attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from walnut.layers.cache import Cache


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class ConvState(Cache):
    """Static, pre-allocated conv window + delta-rule recurrent state.

    Written in place, like `KVCache`, so the buffers keep one address for the
    life of the sequence.
    """

    def __init__(
        self,
        max_batch_size: int,
        conv_dim: int,
        conv_kernel_size: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        # Last conv_kernel-1 conv inputs, so decode convs stay causal.
        self.conv = torch.zeros(
            max_batch_size, conv_dim, conv_kernel_size - 1, dtype=dtype, device=device
        )
        # The delta rule accumulates in float32.
        self.recurrent = torch.zeros(
            max_batch_size,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            dtype=torch.float32,
            device=device,
        )
        self.primed = False

    @property
    def empty(self) -> bool:
        """True until a forward pass has written state into the buffers."""
        return not self.primed


def _recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent gated delta rule with q/k L2-norm; returns (out, final_state).

    q/k/v are (B, S, heads, dim); g/beta are (B, S, heads). Runs from
    initial_state (zeros if None).
    """
    initial_dtype = query.dtype
    query = _l2norm(query)
    key = _l2norm(key)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    )

    batch, heads, seq, k_dim = key.shape
    v_dim = value.shape[-1]
    query = query * (1 / (query.shape[-1] ** 0.5))

    out = torch.zeros(batch, heads, seq, v_dim, dtype=value.dtype, device=value.device)
    if initial_state is None:
        state = torch.zeros(
            batch, heads, k_dim, v_dim, dtype=value.dtype, device=value.device
        )
    else:
        state = initial_state
    for i in range(seq):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return out.transpose(1, 2).contiguous().to(initial_dtype), state


class _RMSNormGated(nn.Module):
    """RMS norm with a SiLU gate."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)


class GatedDeltaNet(nn.Module):
    """Linear-attention token mixer: a causal conv feeding a gated delta-rule
    recurrence. Submodule names match the Qwen3.5 checkpoint."""

    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_k_heads = num_key_heads
        self.num_v_heads = num_value_heads
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.key_dim = key_head_dim * num_key_heads
        self.value_dim = value_head_dim * num_value_heads
        self.conv_kernel_size = conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=conv_kernel_dim,
            groups=self.conv_dim,
            padding=conv_kernel_dim - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(num_value_heads))
        self.norm = _RMSNormGated(value_head_dim, eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # All four in-projections read the same hidden state, so they are held
        # as one gemv and split after. ``in_proj_b``/``in_proj_a`` produce 16
        # values each: as their own kernels they cost the launch floor rather
        # than the 32 KB they read.
        self.in_proj_splits = [
            self.conv_dim,
            self.value_dim,
            num_value_heads,
            num_value_heads,
        ]
        self.in_proj = nn.Linear(hidden_size, sum(self.in_proj_splits), bias=False)

    def make_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> ConvState:
        """Fresh recurrent state; conv/state are already fixed-size, so
        ``max_seq_len`` is ignored."""
        return ConvState(
            max_batch_size,
            self.conv_dim,
            self.conv_kernel_size,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            dtype,
            device,
        )

    def forward(
        self, hidden_states: torch.Tensor, cache: ConvState | None = None
    ) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        pad = self.conv_kernel_size - 1

        qkv_pre, z, b, a = self.in_proj(hidden_states).split(
            self.in_proj_splits, dim=-1
        )
        qkv_pre = qkv_pre.transpose(1, 2)
        z = z.reshape(batch, seq, -1, self.head_v_dim)

        decoding = cache is not None and not cache.empty
        if decoding:
            assert cache is not None
            # Prepend cached conv context; unpadded conv yields exactly S outputs.
            # ``cat`` copies, so writing the window back now cannot disturb it.
            conv_in = torch.cat([cache.conv, qkv_pre], dim=-1)
            cache.conv.copy_(conv_in[..., -pad:])
            conv_out = F.conv1d(conv_in, self.conv1d.weight, groups=self.conv_dim)
            mixed_qkv = F.silu(conv_out).transpose(1, 2)
        else:
            mixed_qkv = F.silu(self.conv1d(qkv_pre)[:, :, :seq]).transpose(1, 2)
            if cache is not None:
                cache.conv.copy_(
                    qkv_pre[..., -pad:]
                    if qkv_pre.shape[-1] >= pad
                    else F.pad(qkv_pre, (pad - qkv_pre.shape[-1], 0))
                )

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch, seq, -1, self.head_k_dim)
        key = key.reshape(batch, seq, -1, self.head_k_dim)
        value = value.reshape(batch, seq, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        init = cache.recurrent if decoding else None
        core, state = _recurrent_gated_delta_rule(query, key, value, g, beta, init)
        if cache is not None:
            cache.recurrent.copy_(state)
            cache.primed = True
        core = self.norm(
            core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        return self.out_proj(core.reshape(batch, seq, -1))
