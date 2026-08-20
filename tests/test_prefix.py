"""The prefix tree: what a prompt beginning like an earlier one gets to skip.

Driven by a stand-in whose next token is a function of the *whole* prefix and
nothing else. Its state is the running sum of every token it has seen, which no
position can repair and no chunk boundary can change, and it reads the very
first token back out of the cache page that holds it. So a restored checkpoint
that is off by a token, and a borrowed page that belongs to somebody else, are
both wrong tokens rather than wrong numbers — and a run that reuses a prefix
has to produce exactly what a run that recomputed it produced.
"""

import threading

import torch

from walnut.cache import PAGE_SIZE, CachePool, PrefixCache, StateCache, pages_for
from walnut.layers.attention import Attention
from walnut.sampler import Sampler, SamplingParams
from walnut.scheduler import Request, Scheduler

VOCAB = 64


class _Sum(StateCache):
    """Recurrent state a checkpoint has to carry: one running total per row."""

    def __init__(self, rows: int) -> None:
        self.total = torch.zeros(rows)

    def buffers(self) -> list:
        return [self.total]

    def view(self, start: int, stop: int) -> "_Sum":
        cache = _Sum.__new__(_Sum)
        cache.total = self.total[start:stop]
        cache.primed = self.primed
        return cache


class _PrefixModel:
    """Emits ``(sum of every token seen + the first one) % VOCAB``.

    Two caches, because prefix reuse hands back two things and both have to be
    right: the running total stands in for the recurrent state a checkpoint
    copies, and the real `KVCache` holds the first token in the page a later
    sequence borrows.
    """

    def __init__(self) -> None:
        self.attn = Attention(num_heads=1, num_kv_heads=1, head_dim=8)
        self.sampler = Sampler()
        self.eos_token_id = None

    def make_cache(self, max_batch_size, max_seq_len, pages=None):
        pages = max_batch_size * pages_for(max_seq_len) if pages is None else pages
        return CachePool(
            [
                _Sum(max_batch_size),
                self.attn.make_cache(pages + 1, torch.float32, None),
            ],
            max_batch_size,
            max_seq_len,
            pages,
        )

    def __call__(self, input_ids, positions, cache):
        rows, seq = input_ids.shape
        batch = cache.batch(positions)
        value = input_ids.float()[..., None, None].expand(rows, seq, 1, 8)
        cache[1].write(batch, torch.zeros_like(value), value.clone())
        cache[0].total += input_ids.float().sum(-1)
        # The first token of each row, read back out of whichever page is
        # holding it — this row's own, or one it was lent.
        first = cache[1].v.reshape(-1, 1, 8)[
            cache.block_table[:, 0].long() * PAGE_SIZE, 0, 0
        ]
        token = (cache[0].total.long() + first.long()) % VOCAB
        logits = torch.zeros(rows, seq, VOCAB)
        logits[:, -1] = torch.nn.functional.one_hot(token, VOCAB).float()
        return logits


def _scheduler(max_batch_size=2, max_seq_len=4096, checkpoints=4, kv_tokens=None):
    return Scheduler(
        _PrefixModel(),
        torch.device("cpu"),
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        cuda_graph=False,
        compile=False,
        kv_tokens=kv_tokens,
        prefix_checkpoints=checkpoints,
    )


def _prompt(tokens: list[int], new: int = 3) -> Request:
    return Request(
        prompt=torch.tensor([tokens]),
        params=SamplingParams(max_new_tokens=new, temperature=0.0),
        stop_ids=frozenset(),
    )


def _run(scheduler, tokens, new=3) -> list[int]:
    """One request, start to finish, with nothing else in flight."""
    request = _prompt(tokens, new)
    scheduler.submit(request)
    return list(request.stream())


def _tokens(length: int, seed: int = 0) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (length,), generator=generator).tolist()


def test_a_repeated_prompt_is_reused_from_the_third_request():
    """First inserts the pages, second marks the boundary worth keeping and
    checkpoints its state going past, third starts there.

    Third and not second on purpose: a checkpoint costs more memory than the
    pages it skips, so the tree waits for a prefix to be shared twice before
    paying for it.
    """
    prompt = _tokens(600)
    scheduler = _scheduler()
    try:
        first = _run(scheduler, prompt)
        assert scheduler.prefix.hits == 0

        second = _run(scheduler, prompt)
        assert scheduler.prefix.hits == 0
        assert len(scheduler.prefix.checkpoints) == 1

        third = _run(scheduler, prompt)
        assert scheduler.prefix.hits == 1
        assert scheduler.prefix.tokens_saved == 2 * PAGE_SIZE
    finally:
        scheduler.close()
    assert first == second == third


def test_reuse_is_only_as_deep_as_the_prompts_agree():
    """A shared head and a divergent tail: the tokens after the split have to
    come out as if nothing had been shared."""
    head = _tokens(600, seed=1)
    one, two = head + _tokens(200, seed=2), head + _tokens(200, seed=3)
    scheduler = _scheduler()
    try:
        _run(scheduler, one)
        _run(scheduler, two)  # marks the shared head, checkpoints it
        warm_one = _run(scheduler, one)
        warm_two = _run(scheduler, two)
        assert scheduler.prefix.hits == 2
    finally:
        scheduler.close()

    cold = _scheduler(checkpoints=0)
    try:
        assert warm_one == _run(cold, one)
        assert warm_two == _run(cold, two)
        assert cold.prefix.hits == 0
    finally:
        cold.close()


def test_a_prompt_is_never_skipped_whole():
    """An exactly repeated prompt still runs its last page. A sequence that
    skipped every token would have no logits to sample its first one from.

    Two pages of prompt, so at most one of them is ever skippable however many
    times it repeats. The checkpoint at the far end is not wasted — a longer
    prompt starting with these 512 tokens can stand on it — but this one has to
    stop short of it.
    """
    prompt = _tokens(512)
    scheduler = _scheduler()
    try:
        runs = [_run(scheduler, prompt) for _ in range(4)]
        assert len(set(map(tuple, runs))) == 1
        assert scheduler.prefix.tokens_saved == PAGE_SIZE
    finally:
        scheduler.close()


def test_without_checkpoints_nothing_is_reused():
    """Pages alone buy nothing on a model with recurrent state, so a store of
    zero checkpoints is a prefix cache that never hits — and still answers."""
    prompt = _tokens(600)
    scheduler = _scheduler(checkpoints=0)
    try:
        assert _run(scheduler, prompt) == _run(scheduler, prompt)
        assert scheduler.prefix.hits == 0
        assert len(scheduler.prefix.checkpoints) == 0
    finally:
        scheduler.close()


def test_a_cached_prefix_yields_to_a_request_that_needs_the_pages():
    """The tree holds pages that are not free. A request arriving when none are
    left must still be admitted, by evicting what nobody is reading."""
    # Four pages of cache, and prompts that want three apiece: the second
    # cannot be served without dropping what the first left behind.
    scheduler = _scheduler(max_batch_size=1, kv_tokens=4 * PAGE_SIZE)
    try:
        for seed in range(4):
            assert _run(scheduler, _tokens(600, seed=seed))
        assert scheduler.pool.free_pages + scheduler.prefix.nodes <= 4
    finally:
        scheduler.close()


def test_concurrent_requests_over_one_shared_prefix_agree():
    """Two sequences reading the same borrowed pages at once, while a third
    writes its own. A refcount that let either page be evicted or reused would
    show up as a wrong token in one of them."""
    head = _tokens(600, seed=7)
    prompts = [head + _tokens(100, seed=s) for s in (8, 9, 10)]
    scheduler = _scheduler(max_batch_size=4)
    alone = _scheduler(max_batch_size=4, checkpoints=0)
    try:
        for prompt in prompts:  # warm the shared head
            _run(scheduler, prompt)
            _run(scheduler, prompt)
        expected = [_run(alone, prompt) for prompt in prompts]

        out: dict[int, list[int]] = {}

        def drive(index, prompt):
            out[index] = _run(scheduler, prompt)

        threads = [
            threading.Thread(target=drive, args=(i, p)) for i, p in enumerate(prompts)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert [out[i] for i in range(len(prompts))] == expected
    finally:
        scheduler.close()
        alone.close()


def test_a_grafted_page_leaves_the_free_list_and_a_redundant_one_returns():
    """Directly, at the tree. A sequence reserves pages for prompt *and*
    completion; the ones covering whole pages of the prompt become the tree's
    when it retires, and the rest go back."""
    pool = _PrefixModel().make_cache(2, 4096)
    tree = PrefixCache(pool, checkpoints=2)
    tokens = _tokens(2 * PAGE_SIZE)

    place = tree.acquire(tokens, len(tokens) + 8)
    assert place is not None and place.prefilled == 0
    # Three pages: two whole ones of prompt, and one for what it will generate.
    assert len(pool.held(place.row)) == 3
    free = pool.free_pages

    tree.release(place.row, tokens)
    assert tree.nodes == 2  # the two whole pages stayed
    assert pool.free_pages == free + 1  # the third came back


def test_a_page_being_read_is_not_evicted():
    """Eviction takes the least recently used leaf *that nobody holds*. Taking
    one out from under a live sequence would leave its block table naming cache
    that is no longer there."""
    pool = _PrefixModel().make_cache(2, 4096, pages=4)
    tree = PrefixCache(pool, checkpoints=2)
    tokens = _tokens(2 * PAGE_SIZE)

    # Build the prefix, then warm it until a checkpoint exists to borrow from.
    for _ in range(3):
        place = tree.acquire(tokens, len(tokens) + 8)
        assert place is not None
        if place.checkpoint_at is not None:
            tree.checkpoint(place.row, place.checkpoint_at)
        tree.release(place.row, tokens)

    holder = tree.acquire(tokens, len(tokens) + 8)
    assert holder is not None and holder.prefilled == PAGE_SIZE
    borrowed = tree._live[holder.row].path[0]
    assert borrowed.refs == 1

    # A second sequence with nothing in common, on a pool with no room left.
    other = tree.acquire(_tokens(2 * PAGE_SIZE, seed=99), 2 * PAGE_SIZE + 8)
    assert other is None or borrowed.refs == 1
    assert borrowed.page not in pool._free_pages
