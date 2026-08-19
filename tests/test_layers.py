import math
from typing import Any, cast

import pytest
import torch

from walnut.layers.attention import Attention, KVCache
from walnut.layers.linear_attention import (
    _CHUNK,
    GatedDeltaNet,
    _chunked_gated_delta_rule,
    _recurrent_gated_delta_rule,
)
from walnut.layers.norm import RMSNorm
from walnut.layers.rotary import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)


def test_rotate_half_swaps_and_negates():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert rotate_half(x).tolist() == [-3.0, -4.0, 1.0, 2.0]


def test_rmsnorm_gives_unit_rms_with_zero_weight():
    norm = RMSNorm(4)  # weight initializes to zeros -> scale of (1 + 0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rmsnorm_preserves_input_dtype():
    norm = RMSNorm(4)
    x = torch.randn(2, 4, dtype=torch.float16)
    assert norm(x).dtype == torch.float16


def test_rmsnorm_matches_manual_formula():
    # Reference computed independently: x / sqrt(mean(x^2) + eps).
    x = [1.0, 2.0, 3.0, 4.0]
    scale = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + 1e-6)
    expected = torch.tensor([[v * scale for v in x]])
    got = RMSNorm(4)(torch.tensor([x]))  # weight is zeros -> factor (1 + 0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_rmsnorm_applies_one_centered_weight():
    # Output scales by (1 + weight), not weight; a zero weight is a no-op gain.
    x = [1.0, 2.0, 3.0, 4.0]
    scale = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + 1e-6)
    weight = [0.0, 1.0, 2.0, 3.0]
    expected = torch.tensor(
        [[v * scale * (1 + w) for v, w in zip(x, weight, strict=True)]]
    )
    norm = RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor(weight))
    assert torch.allclose(norm(torch.tensor([x])), expected, atol=1e-5)


def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=8, rope_theta=10000.0)
    cos, sin = rope(torch.zeros(1, dtype=torch.long))
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_rot, q, atol=1e-5)
    assert torch.allclose(k_rot, k, atol=1e-5)


def test_rope_rotates_pair_by_expected_angle():
    # With head_dim=4, inv_freq[0] == 1, so position m rotates the (dim 0, dim 2)
    # pair by exactly m radians. Feeding the unit vector (1,0) into that pair must
    # yield (cos m, sin m) -- an independent 2D-rotation reference.
    m = 1
    rope = RotaryEmbedding(head_dim=4, rope_theta=10000.0)
    cos, sin = rope(torch.tensor([m]))
    q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])  # (B, S, H, D)
    rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)
    assert rotated[0, 0, 0, 0].item() == pytest.approx(math.cos(m), abs=1e-6)
    assert rotated[0, 0, 0, 2].item() == pytest.approx(math.sin(m), abs=1e-6)
    # A rotation preserves length.
    assert rotated.norm().item() == pytest.approx(1.0, abs=1e-6)


def test_rope_partial_rotary_passes_tail_through():
    # Only the leading rotary_dim (head_dim * factor) dims are rotated.
    rope = RotaryEmbedding(head_dim=8, rope_theta=10000.0, partial_rotary_factor=0.5)
    cos, sin = rope(torch.arange(3))
    rotary_dim = cos.shape[-1]
    assert rotary_dim == 4
    q = torch.randn(1, 3, 2, 8)
    k = torch.randn(1, 3, 2, 8)
    q_rot, _ = apply_rotary_pos_emb(q, k, cos, sin)
    # The passthrough tail is untouched.
    assert torch.allclose(q_rot[..., rotary_dim:], q[..., rotary_dim:])
    # The rotated head is changed at nonzero positions.
    assert not torch.allclose(q_rot[:, 1:, :, :rotary_dim], q[:, 1:, :, :rotary_dim])


#: Attention runs through FlashAttention's variable-length kernel, which is
#: CUDA-only and built for float16/bfloat16 — and it is the only path there is,
#: with no cacheless branch left to check it against on CPU. So every test of
#: attention itself skips on CI's runner. What still runs there is the cache's
#: own bookkeeping (`KVCache.update`, the slot views) and the scheduler driving
#: a stand-in model, which is where the batching logic lives.
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="attention needs CUDA"
)

#: Where and in what precision the kernel exists.
_HALF: Any = {"device": "cuda", "dtype": torch.bfloat16}


def _kv(rows: int, heads: int, length: int, head_dim: int) -> KVCache:
    return KVCache(
        max_batch_size=rows,
        n_kv_heads=heads,
        max_seq_len=length,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )


@cuda_only
def test_attention_matches_manual_softmax():
    """Compare against a hand-rolled causal softmax so the numbers -- not just
    the shapes -- are pinned.

    With no cacheless branch left, this is the only place attention is checked
    against its own definition rather than against another run of the same
    kernel. head_dim is 8 because that is the narrowest the kernel takes; the
    trailing dimensions are zero, so the arithmetic is still the 2-D one.
    """
    head_dim = 8
    attn = Attention(num_heads=1, num_kv_heads=1, head_dim=head_dim)

    def pad(pair):
        return [*pair, *([0.0] * (head_dim - 2))]

    q = torch.tensor([[[pad([1.0, 0.0])], [pad([0.0, 1.0])]]], **_HALF)
    k = torch.tensor([[[pad([1.0, 0.0])], [pad([1.0, 1.0])]]], **_HALF)
    v = torch.tensor([[[pad([2.0, 3.0])], [pad([4.0, 5.0])]]], **_HALF)

    qm, km, vm = (t[0, :, 0].float() for t in (q, k, v))
    scores = (qm @ km.T) * head_dim**-0.5
    ones = torch.ones(2, 2, device="cuda")
    scores = scores.masked_fill(~torch.tril(ones).bool(), float("-inf"))
    expected = torch.softmax(scores, dim=-1) @ vm

    got = attn(q, k, v, _kv(1, 1, 2, head_dim), torch.arange(2, device="cuda"))
    assert torch.allclose(got[0, :, 0].float(), expected, atol=2e-2)


@cuda_only
def test_attention_incremental_matches_prefill():
    """A prompt read at once, and the same tokens stepped through one at a
    time, have to land in the same place -- which is what the cache's positions
    and the kernel's per-row lengths are together for."""
    torch.manual_seed(0)
    attn = Attention(num_heads=4, num_kv_heads=2, head_dim=64)
    seq = 5
    q = torch.randn(1, seq, 4, 64, **_HALF)
    k = torch.randn(1, seq, 2, 64, **_HALF)
    v = torch.randn(1, seq, 2, 64, **_HALF)

    full = attn(q, k, v, _kv(1, 2, seq, 64), torch.arange(seq, device="cuda"))

    cache = _kv(1, 2, seq, 64)
    steps = [
        attn(
            q[:, i : i + 1],
            k[:, i : i + 1],
            v[:, i : i + 1],
            cache,
            input_pos=torch.tensor([[i]], device="cuda"),
        )
        for i in range(seq)
    ]
    incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full.float(), incremental.float(), atol=2e-2)


def _delta_net() -> GatedDeltaNet:
    torch.manual_seed(0)
    return GatedDeltaNet(
        hidden_size=16,
        num_key_heads=2,
        num_value_heads=2,
        key_head_dim=4,
        value_head_dim=4,
        conv_kernel_dim=4,
    )


def test_gated_delta_net_incremental_matches_prefill():
    net = _delta_net()
    seq = 6
    x = torch.randn(1, seq, 16)

    full = net(x)

    cache = net.make_cache(1, seq, torch.float32, None)
    steps = [net(x[:, i : i + 1], cache) for i in range(seq)]
    incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-5)


@pytest.mark.parametrize("seq", [1, 5, _CHUNK, _CHUNK + 1, 2 * _CHUNK + 7])
def test_chunked_delta_rule_matches_the_recurrent_one(seq):
    """The chunked form is the recurrent one reassociated, so it must agree.

    Both are driven directly, past `_gated_delta_rule`'s dispatch, so the
    lengths where only one of them normally runs are covered too. `g` is
    negative and the keys are L2-normalized, as the layer guarantees.
    """
    torch.manual_seed(0)
    heads, k_dim, v_dim = 3, 8, 8
    shape = (1, heads, seq)

    def norm(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    query = norm(torch.randn(*shape, k_dim))
    key = norm(torch.randn(*shape, k_dim))
    value = torch.randn(*shape, v_dim)
    g = -torch.rand(*shape)
    beta = torch.rand(*shape)
    state = torch.randn(1, heads, k_dim, v_dim) * 0.1

    out, final = _recurrent_gated_delta_rule(query, key, value, g, beta, state)
    chunk_out, chunk_final = _chunked_gated_delta_rule(
        query, key, value, g, beta, state
    )

    assert torch.allclose(out, chunk_out, atol=1e-5)
    assert torch.allclose(final, chunk_final, atol=1e-5)


def test_delta_rule_does_not_mutate_the_state_it_is_given():
    """`ConvState.recurrent` is passed in directly and copied out afterwards;
    writing through it would corrupt the cache mid-step."""
    torch.manual_seed(0)
    query, key, value = (torch.randn(1, 2, 5, 4) for _ in range(3))
    g, beta = -torch.rand(1, 2, 5), torch.rand(1, 2, 5)
    state = torch.randn(1, 2, 4, 4) * 0.1
    original = state.clone()

    _recurrent_gated_delta_rule(query, key, value, g, beta, state)
    assert torch.equal(state, original)
    _chunked_gated_delta_rule(query, key, value, g, beta, state)
    assert torch.equal(state, original)


def test_gated_delta_net_writes_cache_buffers_in_place():
    # A captured CUDA graph replays into the buffers it recorded, so the state
    # must stay at one address rather than being rebound to a fresh tensor.
    net = _delta_net()
    cache = net.make_cache(1, 8, torch.float32, None)
    conv, recurrent = cache.conv, cache.recurrent

    assert cache.empty
    net(torch.randn(1, 3, 16), cache)
    assert not cache.empty
    net(torch.randn(1, 1, 16), cache)

    assert cache.conv is conv
    assert cache.recurrent is recurrent
    assert recurrent.abs().sum() > 0


@cuda_only
def test_attention_is_causal_in_prefill():
    torch.manual_seed(0)
    attn = Attention(num_heads=2, num_kv_heads=2, head_dim=64)
    seq = 4
    positions = torch.arange(seq, device="cuda")
    q = torch.randn(1, seq, 2, 64, **_HALF)
    k = torch.randn(1, seq, 2, 64, **_HALF)
    v = torch.randn(1, seq, 2, 64, **_HALF)

    out = attn(q, k, v, _kv(1, 2, seq, 64), positions)
    # Perturbing a future key/value must not change an earlier query's output.
    k2, v2 = k.clone(), v.clone()
    k2[:, -1] += 5.0
    v2[:, -1] += 5.0
    out2 = attn(q, k2, v2, _kv(1, 2, seq, 64), positions)
    assert torch.allclose(out[:, 0].float(), out2[:, 0].float(), atol=2e-2)
    assert not torch.allclose(out[:, -1].float(), out2[:, -1].float(), atol=2e-2)


def _pool(rows: int = 4, length: int = 8) -> KVCache:
    return KVCache(max_batch_size=rows, n_kv_heads=2, max_seq_len=length, head_dim=8)


def test_a_slot_view_writes_the_pool_it_came_from():
    """A prefill runs batch-1 against a view; the batched decode reads the
    pool. If the view were a copy, the sequence would decode from nothing."""
    pool = _pool()
    view = cast(KVCache, pool.slot(2))
    keys = torch.randn(1, 3, 2, 8)
    view.update(torch.arange(3), keys, keys)
    assert torch.equal(pool.k[2, :3], keys[0])
    assert (pool.k[0] == 0).all() and (pool.k[1] == 0).all()


def test_resetting_a_slot_leaves_the_others_alone():
    pool = _pool()
    pool.k.fill_(1.0)
    pool.reset(1)
    assert (pool.k[1] == 0).all()
    assert (pool.k[0] == 1).all() and (pool.k[2] == 1).all()


def test_a_batched_update_writes_one_position_per_row():
    """The bug this guards: rows of a decode batch sit at different positions,
    and a shared-position write would put every row's token in one slot."""
    pool = _pool(rows=3, length=8)
    positions = torch.tensor([[0], [4], [7]])
    values = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1)
    values = values.expand(3, 1, 2, 8).contiguous()
    pool.update(positions, values, values)
    for row, position in enumerate((0, 4, 7)):
        assert (pool.k[row, position] == row).all()
        assert pool.k[row].sum() == pool.k[row, position].sum()


def test_a_conv_state_view_stays_primed():
    """The bug this guards: a view that reported itself empty would send a
    decode step down the prefill branch and restart the recurrence."""
    from walnut.layers.linear_attention import ConvState

    state = ConvState(
        max_batch_size=2,
        conv_dim=4,
        conv_kernel_size=3,
        num_value_heads=2,
        key_head_dim=4,
        value_head_dim=4,
    )
    assert state.empty
    state.prime()
    assert not state.empty and not state.view(0, 1).empty
