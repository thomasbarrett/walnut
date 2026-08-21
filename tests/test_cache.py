"""Cache specs: that a declared size is the size that gets allocated."""

from __future__ import annotations

import pytest
import torch

from walnut.cache import PAGE_SIZE, CachePool, nbytes, pages_that_fit
from walnut.layers import Attention, GatedDeltaNet


def _attention() -> Attention:
    return Attention(num_heads=8, num_kv_heads=2, head_dim=64)


def _linear() -> GatedDeltaNet:
    return GatedDeltaNet(
        hidden_size=256,
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=32,
        value_head_dim=32,
        conv_kernel_dim=4,
    )


def _allocated(cache) -> int:
    return sum(b.numel() * b.element_size() for b in cache.buffers())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layer", [_attention, _linear])
def test_a_spec_declares_what_it_allocates(layer, dtype):
    """The point of a spec is to be asked before allocating, so the two have to
    agree. This is the drift a restated spec cannot rule out and a spec written
    next to its own buffer can."""
    spec = layer().cache_spec()
    rows, pages = 4, 7
    assert spec.nbytes(rows, pages, dtype) == _allocated(
        spec.build(rows, pages, dtype, None)
    )


def test_key_value_storage_is_sized_by_pages_and_not_by_rows():
    """Which is the whole of what paging bought: the pool belongs to no row."""
    spec = _attention().cache_spec()
    assert spec.nbytes(1, 8, torch.float32) == spec.nbytes(64, 8, torch.float32)
    assert spec.nbytes(4, 16, torch.float32) == 2 * spec.nbytes(4, 8, torch.float32)


def test_recurrent_state_is_sized_by_rows_and_not_by_pages():
    """A fixed summary per sequence, however long the sequence runs."""
    spec = _linear().cache_spec()
    assert spec.nbytes(4, 0, torch.float32) == spec.nbytes(4, 4096, torch.float32)
    assert spec.nbytes(8, 0, torch.float32) == 2 * spec.nbytes(4, 0, torch.float32)


def test_pages_that_fit_is_the_largest_pool_inside_the_budget():
    specs = [_attention().cache_spec(), _linear().cache_spec()]
    rows, dtype = 4, torch.bfloat16
    budget = 8 << 20
    pages = pages_that_fit(specs, rows, budget, dtype)
    assert pages > 0
    assert nbytes(specs, rows, pages, dtype) <= budget
    assert nbytes(specs, rows, pages + 1, dtype) > budget


def test_pages_that_fit_reports_zero_rather_than_a_pool_that_does_not_fit():
    """Zero is a real answer — the weights left no room — and the caller is
    expected to fail on it rather than allocate a pool nobody can use."""
    specs = [_attention().cache_spec(), _linear().cache_spec()]
    assert pages_that_fit(specs, 4, 1024, torch.float32) == 0


def test_fixed_cost_does_not_scale_with_pages():
    """The bisection `pages_that_fit` does is only valid because cost is linear
    in pages, so the row-sized part has to stay out of the slope."""
    specs = [_attention().cache_spec(), _linear().cache_spec()]
    rows, dtype = 4, torch.float32
    fixed = nbytes(specs, rows, 0, dtype)
    per_page = nbytes(specs, rows, 1, dtype) - fixed
    assert fixed > 0
    for pages in (1, 9, 100):
        assert nbytes(specs, rows, pages, dtype) == fixed + per_page * pages


def test_a_pool_built_from_specs_holds_the_pages_it_was_asked_for():
    specs = [_attention().cache_spec(), _linear().cache_spec()]
    pool = CachePool.from_specs(specs, 4, 4 * PAGE_SIZE, 8, torch.float32, None)
    assert pool.pages == 8
    assert pool.free_pages == 8
    assert pool.free_rows == 4
    # One page more than it hands out, for `CachePool.SCRATCH`.
    assert _allocated(pool.caches[0]) == specs[0].nbytes(4, 9, torch.float32)
