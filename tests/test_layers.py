import math
from typing import Any, cast

import pytest
import torch

from walnut.cache import PAGE_SIZE, CachePool, pages_for
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
#: own bookkeeping (`KVCache.write`, the slot views) and the scheduler driving
#: a stand-in model, which is where the batching logic lives.
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="attention needs CUDA"
)

#: Where and in what precision the kernel exists.
_HALF: Any = {"device": "cuda", "dtype": torch.bfloat16}


def _pool(
    rows: int,
    heads: int,
    length: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> CachePool:
    """A one-layer paged pool with every row reserved for ``length`` tokens.

    Reserved rather than hand-built, so the block table these tests address
    through is the one `CachePool.reserve` writes — which is what makes the
    scattered-page cases below real rather than staged.
    """
    pages = rows * pages_for(length)
    kv = KVCache(pages + 1, heads, head_dim, dtype, device)
    pool = CachePool([kv], rows, length, pages)
    for _ in range(rows):
        assert pool.reserve(length) is not None
    return pool


def _kv(pool: CachePool) -> KVCache:
    """The pool's one layer, as the cache a pass writes."""
    return cast(KVCache, pool[0])


def _cuda_pool(rows: int, heads: int, length: int, head_dim: int) -> CachePool:
    return _pool(rows, heads, length, head_dim, torch.bfloat16, "cuda")


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

    pool = _cuda_pool(1, 1, 2, head_dim)
    got = attn(q, k, v, _kv(pool), pool.batch(torch.arange(2, device="cuda")))
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

    prefill = _cuda_pool(1, 2, seq, 64)
    full = attn(q, k, v, _kv(prefill), prefill.batch(torch.arange(seq, device="cuda")))

    pool = _cuda_pool(1, 2, seq, 64)
    steps = [
        attn(
            q[:, i : i + 1],
            k[:, i : i + 1],
            v[:, i : i + 1],
            _kv(pool),
            pool.batch(torch.tensor([[i]], device="cuda")),
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

    cache = net.make_cache(1, torch.float32, None)
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
    cache = net.make_cache(1, torch.float32, None)
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

    one, two = _cuda_pool(1, 2, seq, 64), _cuda_pool(1, 2, seq, 64)
    out = attn(q, k, v, _kv(one), one.batch(positions))
    # Perturbing a future key/value must not change an earlier query's output.
    k2, v2 = k.clone(), v.clone()
    k2[:, -1] += 5.0
    v2[:, -1] += 5.0
    out2 = attn(q, k2, v2, _kv(two), two.batch(positions))
    assert torch.allclose(out[:, 0].float(), out2[:, 0].float(), atol=2e-2)
    assert not torch.allclose(out[:, -1].float(), out2[:, -1].float(), atol=2e-2)


def test_a_row_view_writes_the_pool_it_came_from():
    """A prefill runs batch-1 against a row's view; the batched decode reads
    the pool. If the view were a copy, the sequence would decode from
    nothing."""
    pool = _pool(rows=4, heads=2, length=8, head_dim=8)
    view = pool.row(2)
    keys = torch.randn(1, 3, 2, 8)
    cast(KVCache, view[0]).write(view.batch(torch.arange(3)), keys, keys)

    page = int(pool.block_table[2, 0])
    kv = _kv(pool)
    assert torch.equal(kv.k[page, :3], keys[0])
    # Every other row got its own page, and none of them were touched.
    others = {int(pool.block_table[row, 0]) for row in (0, 1, 3)}
    assert page not in others
    assert all((kv.k[other] == 0).all() for other in others)


def test_two_rows_never_share_a_page():
    """The invariant the free list exists for. A page handed out twice would
    have one sequence reading the other's keys, which no length or block table
    downstream could catch."""
    pool = _pool(rows=4, heads=2, length=3 * PAGE_SIZE, head_dim=8)
    held = [{int(page) for page in pool.block_table[row].tolist()} for row in range(4)]
    assert all(len(pages) == 3 for pages in held)
    assert len(set().union(*held)) == 12


def test_a_released_row_hands_its_pages_back():
    """And the next sequence gets them, which is the whole of what paging
    buys: memory returns to the pool at the granularity it was taken."""
    pool = _pool(rows=2, heads=2, length=PAGE_SIZE, head_dim=8)
    assert pool.free_pages == 0 and pool.free_rows == 0
    pool.release(0)
    assert pool.free_pages == 1 and pool.free_rows == 1
    assert pool.reserve(PAGE_SIZE) == 0
    assert pool.free_pages == 0


def test_reserving_more_than_the_pool_holds_is_refused():
    """Refused rather than partly served: a sequence half of whose pages exist
    would run into another's the moment it passed them."""
    pool = _pool(rows=2, heads=2, length=PAGE_SIZE, head_dim=8)
    pool.release(0)
    assert pool.reserve(2 * PAGE_SIZE) is None
    # And the failed attempt gave nothing away.
    assert pool.free_pages == 1 and pool.free_rows == 1


def test_an_idle_row_writes_to_scratch_and_not_to_a_live_page():
    """The bug this guards, which cost a real run its determinism: a decode
    step runs every row of its bucket, and a row holding no sequence has a
    zeroed block table. Addressing page 0, every idle row in the batch wrote a
    key over the *first token* of whichever sequence held that page — visible
    only to that sequence's own attention, and only sometimes, depending on
    who the free list had handed page 0 to."""
    pool = _pool(rows=4, heads=2, length=8, head_dim=8)
    for row in range(1, 4):
        pool.release(row)

    kv = _kv(pool)
    live = int(pool.block_table[0, 0])
    assert live != CachePool.SCRATCH
    kv.k[live, 0] = 7.0

    # One decode step over the whole bucket: row 0 live at position 1, and
    # three rows nobody holds, all of them sitting at position 0.
    positions = torch.tensor([[1], [0], [0], [0]])
    kv.write(pool.batch(positions), torch.ones(4, 1, 2, 8), torch.ones(4, 1, 2, 8))

    assert (kv.k[live, 0] == 7.0).all()
    assert (kv.k[CachePool.SCRATCH, 0] == 1.0).all()


def test_a_batched_update_writes_one_position_per_row():
    """The bug this guards: rows of a decode batch sit at different positions,
    and a shared-position write would put every row's token in one cell."""
    pool = _pool(rows=3, heads=2, length=8, head_dim=8)
    positions = torch.tensor([[0], [4], [7]])
    values = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1)
    values = values.expand(3, 1, 2, 8).contiguous()
    _kv(pool).write(pool.batch(positions), values, values)

    kv = _kv(pool)
    for row, position in enumerate((0, 4, 7)):
        page = int(pool.block_table[row, 0])
        assert (kv.k[page, position] == row).all()
        assert kv.k[page].sum() == kv.k[page, position].sum()


def test_a_row_spanning_pages_writes_across_them():
    """A position past the first page must land in the row's *second* page,
    wherever the free list put it -- which is the one thing a stride could not
    express."""
    pool = _pool(rows=2, heads=2, length=2 * PAGE_SIZE, head_dim=8)
    positions = torch.tensor([[PAGE_SIZE - 1, PAGE_SIZE]])
    values = torch.ones(1, 2, 2, 8)
    view = pool.row(1)
    cast(KVCache, view[0]).write(view.batch(positions), values, values)

    kv, first, second = _kv(pool), *pool.block_table[1].tolist()
    assert (kv.k[int(first), PAGE_SIZE - 1] == 1).all()
    assert (kv.k[int(second), 0] == 1).all()
    assert kv.k.sum() == 2 * 2 * 8


def test_a_conv_state_view_stays_primed():
    """The bug this guards: a view that reported itself empty would send a
    decode step down the prefill branch and restart the recurrence."""
    from walnut.layers.linear_attention import ConvState

    state = ConvState(
        rows=2,
        conv_dim=4,
        conv_kernel_size=3,
        num_value_heads=2,
        key_head_dim=4,
        value_head_dim=4,
    )
    assert state.empty
    state.prime()
    assert not state.empty and not state.view(0, 1).empty
