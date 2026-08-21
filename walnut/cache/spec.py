"""What a layer's cache needs, stated before any of it is allocated.

`Cache` is storage; a spec is the promise of storage. The distinction only
starts to matter when something has to *choose* how much to allocate, which is
exactly the question a pool cannot answer by building one and looking: on a
16 GB card the answer might be "less than you asked for", and finding that out
by allocating is how a server dies at start-up rather than starting smaller.

A spec is written by the layer that will read the cache, next to the buffer it
describes and the kernel that walks it, so the two cannot drift. What is *not*
here is a spec hierarchy — no registry, no merge lattice, no per-architecture
subclass to look up. `walnut.layers.attention` and
`walnut.layers.linear_attention` each define the one spec they need and hand it
up. An engine whose layers do not own their caches has to reconstruct all of
this; walnut's do, so this file is a protocol and a bisection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from walnut.cache.state import Cache


class CacheSpec(Protocol):
    """One layer's cache, as a size and a way to build it.

    ``rows`` is how many sequences may be in flight, ``pages`` how much
    token-addressed storage the pool holds in total. A spec uses whichever of
    the two it is sized by — key/value storage grows with pages, recurrent
    state with rows — and ignores the other.
    """

    def nbytes(self, rows: int, pages: int, dtype: torch.dtype) -> int:
        """Bytes this layer's cache would occupy at that size."""
        ...

    def build(
        self,
        rows: int,
        pages: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> Cache:
        """Allocate it."""
        ...


def nbytes(
    specs: Sequence[CacheSpec], rows: int, pages: int, dtype: torch.dtype
) -> int:
    """What a pool of ``pages`` over ``rows`` would cost, across every layer."""
    return sum(spec.nbytes(rows, pages, dtype) for spec in specs)


def pages_that_fit(
    specs: Sequence[CacheSpec], rows: int, budget: int, dtype: torch.dtype
) -> int:
    """The most pages ``budget`` bytes will hold, at ``rows`` rows.

    Every cost here is linear in ``pages`` — a page is a page, whatever else
    the model is doing — so two evaluations are enough to solve for it, and no
    spec has to publish its own coefficients for a caller to do arithmetic on.
    That keeps the protocol two methods wide however many kinds of state a
    future layer keeps.

    Zero is a real answer, and the caller is expected to treat it as one: it
    means the weights left no room, not that a smaller pool would do.
    """
    fixed = nbytes(specs, rows, 0, dtype)
    per_page = nbytes(specs, rows, 1, dtype) - fixed
    if per_page <= 0:
        # No layer is sized by pages at all. Nothing to solve for.
        return 0
    return max(0, (budget - fixed) // per_page)
