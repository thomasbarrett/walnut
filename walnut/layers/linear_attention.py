"""Gated-DeltaNet linear attention."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from walnut.cache import StateCache
from walnut.layers.linear import FusedLinear

#: Positions per chunk in `_chunked_gated_delta_rule`. The chunk's cost is
#: quadratic in this and the number of sequential steps is inversely
#: proportional to it, so the balance depends on which side the loop is bound
#: by. It is bound by the host: a chunk's matmuls are far too small to fill the
#: GPU, so a prefill costs what it costs to *issue* one chunk times the number
#: of chunks, and prefill time is close to linear in that count. 256 is where
#: measurement puts the turn — a 1024-token prompt takes 91 ms at 64 and 35 ms
#: at 256 — and going further only wins for a prompt no longer than one chunk.
#: The quadratic term stays affordable at this width, and the reassociation
#: costs nothing in accuracy: against the recurrent form, a 1024-position run
#: is 4e-7 relative at this chunk size, three orders inside bfloat16's own
#: resolution. Decode does not come through here at all — a single position
#: takes `_recurrent_gated_delta_rule`.
_CHUNK = 256


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class ConvState(StateCache):
    """Static, pre-allocated conv window + delta-rule recurrent state.

    A fixed summary of the whole sequence rather than a cell per token, which
    is what `StateCache` names: it cannot be split into pages, and two
    sequences sharing a prompt cannot point at one copy of it — the state
    after n tokens is the same for both, but a pointer would also hand the
    second one the right to advance it. Hence the row per sequence, where
    `KVCache` needs none.

    Written in place, like `KVCache`, so the buffers keep one address for the
    life of the sequence.
    """

    def __init__(
        self,
        rows: int,
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
            rows, conv_dim, conv_kernel_size - 1, dtype=dtype, device=device
        )
        # The delta rule accumulates in float32.
        self.recurrent = torch.zeros(
            rows,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            dtype=torch.float32,
            device=device,
        )
        self.primed = False

    def buffers(self) -> list[torch.Tensor]:
        return [self.conv, self.recurrent]

    def view(self, start: int, stop: int) -> ConvState:
        """A view inherits ``primed``: it names the same buffers, so a view that
        called itself empty would take the prefill branch over state the
        sequence had already built and silently start it over.
        """
        state = ConvState.__new__(ConvState)
        state.conv = self.conv[start:stop]
        state.recurrent = self.recurrent[start:stop]
        state.primed = self.primed
        return state


@dataclass(frozen=True)
class ConvStateSpec:
    """`ConvState`, as a size and a way to build it.

    Sized by rows and not by pages, which is the same statement `StateCache`
    makes about sharing: a fixed summary per sequence, however long that
    sequence runs.

    The two halves are not the same dtype. The conv window holds activations
    and follows the model; the delta rule accumulates in float32 whatever the
    model is, so a bf16 model still pays four bytes a cell for it.
    """

    conv_dim: int
    conv_kernel_size: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int

    def nbytes(self, rows: int, pages: int, dtype: torch.dtype) -> int:
        window = self.conv_dim * (self.conv_kernel_size - 1) * dtype.itemsize
        recurrent = (
            self.num_value_heads
            * self.key_head_dim
            * self.value_head_dim
            * torch.float32.itemsize
        )
        return rows * (window + recurrent)

    def build(
        self,
        rows: int,
        pages: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> ConvState:
        return ConvState(
            rows,
            self.conv_dim,
            self.conv_kernel_size,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            dtype,
            device,
        )


def _recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The delta rule stepped one position at a time; returns (out, state).

    Inputs are the normalized (B, heads, S, dim) float32 tensors `_gated_delta_rule`
    prepares. Every step depends on the one before, so this costs S rounds of
    host-side dispatch: it is the decode path, where S is 1 and the whole body
    is captured into `DecodeGraph`. `_chunked_gated_delta_rule` covers prefill.
    """
    batch, heads, seq, _ = key.shape
    v_dim = value.shape[-1]
    out = torch.zeros(batch, heads, seq, v_dim, dtype=value.dtype, device=value.device)
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

    return out, state


@functools.lru_cache(maxsize=8)
def _causal_masks(
    length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(eye, j <= t, j < t)`` for a chunk, built once per length and device."""
    ones = torch.ones(length, length, dtype=torch.bool, device=device)
    eye = torch.eye(length, dtype=torch.float32, device=device)
    return eye, ones.tril(), ones.tril(-1)


def _chunked_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The delta rule over `_CHUNK` positions at a time; returns (out, state).

    Same inputs and outputs as `_recurrent_gated_delta_rule`, and the same
    arithmetic reassociated so a chunk costs a fixed handful of matmuls instead
    of one dispatch round per position. That is what prefill needs: the loop
    body is far too small to keep the GPU busy, so S sequential rounds are paid
    almost entirely in host-side dispatch.

    Writing the step's rank-1 update as ``S_t = a_t S_{t-1} + k_t u_t^T`` makes
    the state linear in ``u``, so with ``c_t = sum_{j<=t} g_j`` a whole chunk
    closes in one solve::

        u_t = b_t [ v_t - e^{c_t} S_0^T k_t - sum_{j<t} e^{c_t-c_j}(k_t.k_j) u_j ]
        o_t = e^{c_t} S_0^T q_t + sum_{j<=t} e^{c_t-c_j} (q_t.k_j) u_j
        S_C = e^{c_C} S_0 + sum_j e^{c_C-c_j} k_j u_j^T

    The ``u`` system is unit lower triangular, hence exactly solvable. Keys are
    L2-normalized and ``g <= 0``, so every coefficient is bounded by 1 and the
    solve stays well conditioned however long the chunk runs.
    """
    outs = []
    for start in range(0, key.shape[-2], _CHUNK):
        window = slice(start, start + _CHUNK)
        q, k, v = query[:, :, window], key[:, :, window], value[:, :, window]
        eye, causal, strict = _causal_masks(k.shape[-2], k.device)

        c = g[:, :, window].cumsum(-1)
        # Masking before the exp zeroes the masked entries for free.
        logit = c.unsqueeze(-1) - c.unsqueeze(-2)
        gain = c.exp().unsqueeze(-1)
        b = beta[:, :, window].unsqueeze(-1)

        coupling = (k @ k.transpose(-1, -2)) * logit.masked_fill(
            ~strict, -torch.inf
        ).exp()
        rhs = b * (v - gain * (k @ state))
        u = torch.linalg.solve_triangular(
            eye + b * coupling, rhs, upper=False, unitriangular=True
        )

        weight = (q @ k.transpose(-1, -2)) * logit.masked_fill(
            ~causal, -torch.inf
        ).exp()
        outs.append(gain * (q @ state) + weight @ u)

        tail = (c[..., -1:] - c).exp().unsqueeze(-1)
        state = gain[..., -1:, :] * state + k.transpose(-1, -2) @ (tail * u)

    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2), state


def _gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta rule with q/k L2-norm; returns (out, final_state).

    q/k/v are (B, S, heads, dim); g/beta are (B, S, heads). Runs from
    initial_state (zeros if None). A single position takes the recurrent form,
    which is what `DecodeGraph` captures; a prompt takes the chunked one.
    """
    initial_dtype = query.dtype
    query = _l2norm(query)
    key = _l2norm(key)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    )

    batch, heads, seq, k_dim = key.shape
    query = query * (1 / (k_dim**0.5))

    if initial_state is None:
        state = torch.zeros(
            batch,
            heads,
            k_dim,
            value.shape[-1],
            dtype=value.dtype,
            device=value.device,
        )
    else:
        state = initial_state

    rule = _recurrent_gated_delta_rule if seq == 1 else _chunked_gated_delta_rule
    out, state = rule(query, key, value, g, beta, state)
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
        self.in_proj = FusedLinear(
            hidden_size,
            {
                "in_proj_qkv": self.conv_dim,
                "in_proj_z": self.value_dim,
                "in_proj_b": num_value_heads,
                "in_proj_a": num_value_heads,
            },
        )

    def cache_spec(self) -> ConvStateSpec:
        """What this layer's recurrent state costs, before it is allocated."""
        return ConvStateSpec(
            self.conv_dim,
            self.conv_kernel_size,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        )

    def forward(
        self, hidden_states: torch.Tensor, cache: ConvState | None = None
    ) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        pad = self.conv_kernel_size - 1

        qkv_pre, z, b, a = self.in_proj(hidden_states)
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
        core, state = _gated_delta_rule(query, key, value, g, beta, init)
        if cache is not None:
            cache.recurrent.copy_(state)
            cache.primed = True
        core = self.norm(
            core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        return self.out_proj(core.reshape(batch, seq, -1))
