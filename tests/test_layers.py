import math

import pytest
import torch

from walnut.layers.attention import Attention, KVCache
from walnut.layers.linear_attention import GatedDeltaNet
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


def test_attention_matches_manual_softmax():
    # Single head, head_dim=2, seq=2: compare against a hand-rolled causal
    # softmax attention so the numbers -- not just the shapes -- are pinned.
    attn = Attention(num_heads=1, num_kv_heads=1, head_dim=2)
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])  # (B=1, S=2, H=1, D=2)
    k = torch.tensor([[[[1.0, 0.0]], [[1.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 3.0]], [[4.0, 5.0]]]])

    qm, km, vm = q[0, :, 0], k[0, :, 0], v[0, :, 0]
    scores = (qm @ km.T) * (2**-0.5)
    scores = scores.masked_fill(~torch.tril(torch.ones(2, 2)).bool(), float("-inf"))
    expected = torch.softmax(scores, dim=-1) @ vm

    got = attn(q, k, v)[0, :, 0]
    assert torch.allclose(got, expected, atol=1e-6)


def test_attention_incremental_matches_prefill():
    torch.manual_seed(0)
    attn = Attention(num_heads=4, num_kv_heads=2, head_dim=8)
    seq = 5
    q = torch.randn(1, seq, 4, 8)
    k = torch.randn(1, seq, 2, 8)
    v = torch.randn(1, seq, 2, 8)

    full = attn(q, k, v)

    cache = KVCache(
        max_batch_size=1, n_kv_heads=2, max_seq_len=seq, head_dim=8, dtype=q.dtype
    )
    steps = [
        attn(
            q[:, i : i + 1],
            k[:, i : i + 1],
            v[:, i : i + 1],
            cache,
            input_pos=torch.tensor([i]),
        )
        for i in range(seq)
    ]
    incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-5)


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


def test_attention_is_causal_in_prefill():
    torch.manual_seed(0)
    attn = Attention(num_heads=2, num_kv_heads=2, head_dim=8)
    seq = 4
    q = torch.randn(1, seq, 2, 8)
    k = torch.randn(1, seq, 2, 8)
    v = torch.randn(1, seq, 2, 8)

    out = attn(q, k, v)
    # Perturbing a future key/value must not change an earlier query's output.
    k2 = k.clone()
    v2 = v.clone()
    k2[:, -1] += 5.0
    v2[:, -1] += 5.0
    out2 = attn(q, k2, v2)
    assert torch.allclose(out[:, 0], out2[:, 0], atol=1e-5)
    assert not torch.allclose(out[:, -1], out2[:, -1], atol=1e-5)
