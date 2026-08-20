"""Decode state: what a sequence remembers, and where a pass finds it.

Three questions are easy to conflate and worth keeping apart, because they
change independently — which is why each has a module rather than a section:

*What* a layer remembers — `walnut.cache.state`. Full attention keeps a key and
a value for every token it has seen, so its state grows with the sequence and
every cell of it stays individually addressable: `TokenCache`. Linear attention
keeps a fixed summary of everything before, so its state is one blob per
sequence no matter how long that sequence runs: `StateCache`. The split is not
cosmetic — only the first kind can be split into blocks, and only the first kind
can be *shared* between two sequences that begin with the same tokens.

*Where* it lives — `walnut.cache.pool`. `CachePool` owns the storage for every
layer at once and hands out placements; `CacheView` narrows it to the rows one
pass runs against. Today a placement is a slot — one row of a preallocated pool,
`max_seq_len` cells wide — and `reserve` is a free list. A paged pool changes
what `reserve` returns and nothing else about who calls it, and a prefix tree
sits above `reserve` handing back placements that overlap.

*How a pass reaches it* — `walnut.cache.batch`. `Batch` is that answer, computed
once per forward and read by every layer: how long each row's context is, and
which cell each token of this pass writes. Deriving those per layer, from the
shape of the position tensor, is what makes a cache layout impossible to change
— the layout ends up restated in every mixer. Here it is stated once, by the
view, which is the only object that knows how sequences are laid out.

The distinction `Batch` draws that a position tensor cannot: a token's
*placement* in the cache is its ordinal in the sequence, while its *rotary
position* is whatever the position encoding says it is. Those agree for text and
part ways under M-RoPE, where an image occupies one cache cell per token but
three interleaved coordinate axes. Rotary reads `positions`; the cache reads
`Batch`.

What is *not* here: the buffers themselves. `KVCache` and `ConvState` live with
the mixers that read them, because a cache's layout and the kernel that walks it
are one decision — see `walnut.layers.attention` and
`walnut.layers.linear_attention`. This package holds the placement of sequences
within those buffers, which is the half that paging and prefix sharing change.
"""

from walnut.cache.batch import Batch
from walnut.cache.pool import CachePool, CacheView
from walnut.cache.state import Cache, StateCache, TokenCache

__all__ = [
    "Batch",
    "Cache",
    "CachePool",
    "CacheView",
    "StateCache",
    "TokenCache",
]
