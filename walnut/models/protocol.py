"""What the serving loop requires of a model, and nothing more.

`Scheduler` and `DecodeGraph` took a ``model: Any``, which is another way of
saying the contract was whatever the one implementation happened to do. It is
two contracts, not one, and they are needed by different callers:

`Forward` is a pass — tokens in, logits out, against a cache. It is all that
CUDA graph capture needs, and deliberately so: what `DecodeGraph` is handed is
usually not a model at all but ``torch.compile(model)``, a bare callable with
no cache to build and no sampler to draw from.

`CausalLM` is a model the scheduler can serve: a pass, plus the two things the
loop cannot supply for itself.

Note what is *not* here. Nothing asks a model what kind of attention it uses,
whether it keeps recurrent state, or how its cache is laid out: `make_cache`
answers all of that by returning a pool already built to suit, and the layers
that will read it are the ones that built it. A serving loop that had to ask
would be a serving loop with a branch per architecture in it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from walnut.cache import CachePool, CacheView
    from walnut.runner.sampling import Sampler


class Forward(Protocol):
    """One pass over ``input_ids``, reading and writing ``cache``.

    ``positions`` is the rotary position of each token, which is not
    necessarily where it lands in the cache — see `walnut.cache.Batch`.
    """

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor,
        cache: CacheView,
    ) -> torch.Tensor: ...


class CausalLM(Forward, Protocol):
    """A model the scheduler can serve."""

    #: Draws the next token from a row of logits. A model's own, because
    #: sampling is where an architecture may need to differ (a draft head, a
    #: constrained decode) and the loop should not have to know that it does.
    sampler: Sampler

    def make_cache(
        self, max_batch_size: int, max_seq_len: int, pages: int | None = None
    ) -> CachePool:
        """Build the decode state for ``max_batch_size`` rows over ``pages``.

        The model decides what its layers need; the caller decides only how
        much of it there is to go around.
        """
        ...
