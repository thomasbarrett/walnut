"""The gated delta rule's decode step, as two Triton kernels.

A single decode position through `walnut.layers.linear_attention.GatedDeltaNet`
is a causal conv, a SiLU, an L2 norm per head and one rank-1 update of the
recurrent state. Inductor compiles that chain into **ten kernels per layer**
costing ~10.2 us, against ~1.3 us of memory traffic: at batch 1 each one moves
a few kilobytes and pays the kernel launch floor to do it. Eighteen linear
layers made that ~14% of walnut's decode step.

The two kernels here are that same arithmetic, in the same order and the same
precisions, in two launches. `_prepare` covers everything up to the delta rule
and `_step` covers the rule itself.

**Why two and not one.** The conv reads each channel's cached window and writes
it forward, so a channel must belong to exactly one program or the shift races.
The rule wants a different decomposition — see `_step` — and the two cannot be
gridded together without one of them reading what the other is still writing.

Prefill does not come through here: its sequence axis is long enough to fill
the GPU on its own, and `_chunked_gated_delta_rule` already covers it. Nor does
CPU, where there is no Triton and nothing to launch.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - Triton ships with the CUDA wheels
    HAVE_TRITON = False


#: Value columns per program in `_step`. The state is (key_dim, value_dim) per
#: head and every value column is independent, so this is the knob that decides
#: how wide the launch is: at batch 1 and 16 heads, 16 columns gives 128 blocks
#: where a whole head per block would give 16 and leave nine tenths of the SMs
#: idle. Measured on an RTX 5090 at 2.89 us/layer against 3.19 at 32 and 5.23
#: at 128; at batch 8, where the grid is wide either way, all three land within
#: 1% of the 10.4 us bandwidth roofline.
_BLOCK_V = 16
_WARPS = 2


def supported(
    key_head_dim: int, value_head_dim: int, num_k_heads: int, num_v_heads: int
) -> bool:
    """Whether `gated_delta_decode` covers this layer's shape.

    Both head dimensions index a Triton block directly, so both must be powers
    of two, and the value heads must partition evenly over the key heads for
    ``h // rep`` to name a key head. Anything else falls back to the PyTorch
    path, which computes the same thing.
    """
    return (
        HAVE_TRITON
        and key_head_dim & (key_head_dim - 1) == 0
        and value_head_dim & (value_head_dim - 1) == 0
        and num_v_heads % num_k_heads == 0
    )


if HAVE_TRITON:

    @triton.jit
    def _conv_tap(qkv_row, conv_row, w_ptr, chan, offs_t, mask_t, KW: tl.constexpr):
        """Depthwise conv over a channel block's cached window, then shift it.

        Returns the pre-activation, rounded through the activation dtype: aten's
        depthwise conv writes that dtype and the fused chain reads it back, so
        rounding here keeps this path numerically identical to the one it
        replaces rather than merely close.

        Every load lands before the store, and a channel belongs to exactly one
        program, so shifting the window in place is safe.
        """
        x = tl.load(qkv_row + chan).to(tl.float32)
        base = conv_row + chan[:, None] * (KW - 1) + offs_t[None, :]
        prev = tl.load(base, mask=mask_t[None, :], other=0)
        w_prev = tl.load(
            w_ptr + chan[:, None] * KW + offs_t[None, :], mask=mask_t[None, :], other=0
        ).to(tl.float32)
        acc = tl.sum(prev.to(tl.float32) * w_prev, axis=1)
        acc += x * tl.load(w_ptr + chan * KW + (KW - 1)).to(tl.float32)

        ahead = offs_t[None, :] + 1
        nxt = tl.load(
            conv_row + chan[:, None] * (KW - 1) + ahead, mask=ahead < KW - 1, other=0
        )
        tl.store(
            base,
            tl.where(ahead < KW - 1, nxt, x[:, None].to(nxt.dtype)),
            mask=mask_t[None, :],
        )
        return acc.to(qkv_row.dtype.element_ty).to(tl.float32)

    @triton.jit
    def _prepare(
        proj_ptr,
        conv_ptr,
        w_ptr,
        alog_ptr,
        dtb_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        s_proj_b,
        s_conv_b,
        GATE: tl.constexpr,
        NK: tl.constexpr,
        NV: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        KW: tl.constexpr,
        KWP: tl.constexpr,
        KEY_DIM: tl.constexpr,
        EPS: tl.constexpr,
    ):
        """conv, SiLU, split, L2 norm and the gates: one head per program.

        Lanes ``0..NK-1`` own the query and key channels of key head ``pid``;
        lanes ``NK..NK+NV-1`` own the value channels of value head ``pid - NK``
        and that head's ``beta`` and decay. Splitting the lanes by *channel*
        rather than by value head is what makes the conv-window shift race-free
        when several value heads share a key head.
        """
        row = tl.program_id(0)
        pid = tl.program_id(1)
        conv_row = conv_ptr + row * s_conv_b
        qkv_row = proj_ptr + row * s_proj_b
        offs_t = tl.arange(0, KWP)
        mask_t = offs_t < KW - 1

        if pid < NK:
            offs_qk = tl.arange(0, DK)
            for part in tl.static_range(2):
                chan = part * KEY_DIM + pid * DK + offs_qk
                pre = _conv_tap(qkv_row, conv_row, w_ptr, chan, offs_t, mask_t, KW)
                y = pre * tl.sigmoid(pre)
                y = y * tl.rsqrt(tl.sum(y * y, axis=0) + EPS)
                dst = (q_ptr if part == 0 else k_ptr) + row * NK * DK + pid * DK
                # The query carries the 1/sqrt(d) scale the rule applies to it.
                tl.store(dst + offs_qk, y * tl.rsqrt(DK * 1.0) if part == 0 else y)
        else:
            h = pid - NK
            offs_v = tl.arange(0, DV)
            chan = 2 * KEY_DIM + h * DV + offs_v
            pre = _conv_tap(qkv_row, conv_row, w_ptr, chan, offs_t, mask_t, KW)
            tl.store(v_ptr + row * NV * DV + h * DV + offs_v, pre * tl.sigmoid(pre))

            gate = proj_ptr + row * s_proj_b + GATE
            aa = tl.load(gate + NV + h).to(tl.float32)
            dtb = tl.load(dtb_ptr + h).to(tl.float32)
            z = aa + dtb
            # softplus, guarded the way torch guards it: past the threshold the
            # correction is below float32 resolution and exp(z) would overflow.
            softplus = tl.where(z > 20.0, z, tl.log(1.0 + tl.exp(z)))
            alog = tl.load(alog_ptr + h).to(tl.float32)
            tl.store(g_ptr + row * NV + h, tl.exp(-tl.exp(alog) * softplus))
            bb = tl.load(gate + h).to(tl.float32)
            tl.store(beta_ptr + row * NV + h, tl.sigmoid(bb))

    @triton.jit
    def _step(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        state_ptr,
        out_ptr,
        s_state_b,
        s_state_h,
        NK: tl.constexpr,
        NV: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BV: tl.constexpr,
        REP: tl.constexpr,
    ):
        """One delta-rule position, split across the value dimension.

        ``S <- gS + k (v - S^T k) beta`` then ``o = S^T q``, which reads and
        writes the whole state and does almost no arithmetic on it -- so the
        only thing that matters is how much of the GPU the launch covers.
        Given ``q`` and ``k``, every value column is independent, so the grid
        carries a third axis over blocks of them and each program owns a
        ``(DK, BV)` slice of one head's state.
        """
        row = tl.program_id(0)
        h = tl.program_id(1)
        kh = h // REP

        offs_k = tl.arange(0, DK)
        offs_v = tl.program_id(2) * BV + tl.arange(0, BV)
        mask_v = offs_v < DV

        q = tl.load(q_ptr + row * NK * DK + kh * DK + offs_k)
        k = tl.load(k_ptr + row * NK * DK + kh * DK + offs_k)
        v = tl.load(v_ptr + row * NV * DV + h * DV + offs_v, mask=mask_v, other=0.0)
        g = tl.load(g_ptr + row * NV + h)
        beta = tl.load(beta_ptr + row * NV + h)

        sp = (
            state_ptr
            + row * s_state_b
            + h * s_state_h
            + offs_k[:, None] * DV
            + offs_v[None, :]
        )
        state = tl.load(sp, mask=mask_v[None, :], other=0.0) * g
        kv_mem = tl.sum(state * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        state = state + k[:, None] * delta[None, :]
        out = tl.sum(state * q[:, None], axis=0)

        tl.store(sp, state, mask=mask_v[None, :])
        tl.store(
            out_ptr + row * NV * DV + h * DV + offs_v,
            out.to(out_ptr.dtype.element_ty),
            mask=mask_v,
        )


@torch.library.custom_op(
    "walnut::gated_delta_decode", mutates_args={"conv_state", "recurrent"}
)
def gated_delta_decode(
    proj: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent: torch.Tensor,
    num_k_heads: int,
    key_head_dim: int,
    gate_offset: int,
    eps: float,
) -> torch.Tensor:
    """One decode position of the gated delta rule; returns the core output.

    ``proj`` is the in-projection's whole output for this step, ``(rows, 1,
    features)``, unsplit: the conv input runs from 0 and the two gate parts
    from ``gate_offset``, which the caller reads off `FusedLinear.offset` so
    the layout stays written down in one place. Handing over the joint buffer
    rather than three views of it is what keeps `torch.compile` from copying
    each one out for an opaque callee -- three kernels a layer, measured.

    ``conv_state`` and ``recurrent`` are the cache's own buffers and are
    advanced in place -- the caller holds them at a fixed address for a
    captured graph to replay into, so returning fresh ones would break the
    capture.

    Returned is ``(rows, value_heads, value_head_dim)`` in ``qkv``'s dtype,
    which is what the layer's gated norm and out-projection take. A custom op
    rather than a bare kernel call so `torch.compile` keeps it in one graph and
    still fuses the elementwise work on either side of it.
    """
    rows = proj.shape[0]
    _, num_v_heads, key_dim, value_head_dim = recurrent.shape
    assert proj.stride(-1) == 1 and conv_state.is_contiguous()
    assert recurrent.stride(-1) == 1 and recurrent.stride(-2) == value_head_dim
    assert key_dim == key_head_dim

    empty = torch.empty
    dev, f32 = proj.device, torch.float32
    q = empty(rows, num_k_heads, key_head_dim, device=dev, dtype=f32)
    k = empty(rows, num_k_heads, key_head_dim, device=dev, dtype=f32)
    v = empty(rows, num_v_heads, value_head_dim, device=dev, dtype=f32)
    g = empty(rows, num_v_heads, device=dev, dtype=f32)
    beta = empty(rows, num_v_heads, device=dev, dtype=f32)
    out = empty(rows, num_v_heads, value_head_dim, device=dev, dtype=proj.dtype)
    kernel_size = conv_weight.shape[-1]

    _prepare[(rows, num_k_heads + num_v_heads)](
        proj,
        conv_state,
        conv_weight,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        proj.stride(0),
        conv_state.stride(0),
        GATE=gate_offset,
        NK=num_k_heads,
        NV=num_v_heads,
        DK=key_head_dim,
        DV=value_head_dim,
        KW=kernel_size,
        KWP=triton.next_power_of_2(kernel_size - 1),
        KEY_DIM=num_k_heads * key_head_dim,
        EPS=eps,
        num_warps=_WARPS,
    )
    block = min(_BLOCK_V, value_head_dim)
    _step[(rows, num_v_heads, triton.cdiv(value_head_dim, block))](
        q,
        k,
        v,
        g,
        beta,
        recurrent,
        out,
        recurrent.stride(0),
        recurrent.stride(1),
        NK=num_k_heads,
        NV=num_v_heads,
        DK=key_head_dim,
        DV=value_head_dim,
        BV=block,
        REP=num_v_heads // num_k_heads,
        num_warps=_WARPS,
    )
    return out


@gated_delta_decode.register_fake
def _(
    proj: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent: torch.Tensor,
    num_k_heads: int,
    key_head_dim: int,
    gate_offset: int,
    eps: float,
) -> torch.Tensor:
    return proj.new_empty(recurrent.shape[0], recurrent.shape[1], recurrent.shape[3])
