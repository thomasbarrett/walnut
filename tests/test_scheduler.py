"""The continuous-batching loop, driven by a model stand-in.

The stand-in generates ``token + 1`` per step out of a real `KVCache`, so a
sequence's output is a function of its own prompt and nothing else: a row that
read another row's pages, or its own at the wrong position, shows up as a
wrong token rather than as a wrong number.
"""

import threading

import pytest
import torch

from walnut.cache import PAGE_SIZE, CachePool, StateCache, pages_for
from walnut.graph import buckets
from walnut.layers.attention import Attention
from walnut.sampler import Sampler, SamplingParams
from walnut.scheduler import Request, RequestError, Scheduler

VOCAB = 64


class _StepModel:
    """Emits ``(last token + 1) % VOCAB``, through a real KV cache."""

    def __init__(self) -> None:
        self.attn = Attention(num_heads=1, num_kv_heads=1, head_dim=8)
        self.sampler = Sampler()
        self.eos_token_id = None
        self.steps = 0
        self.fail = False
        self.crash = False

    def make_cache(self, max_batch_size, max_seq_len, pages=None):
        pages = max_batch_size * pages_for(max_seq_len) if pages is None else pages
        return CachePool(
            [self.attn.make_cache(pages + 1, torch.float32, None)],
            max_batch_size,
            max_seq_len,
            pages,
        )

    def __call__(self, input_ids, positions, cache):
        if self.crash:
            raise BaseException("the loop died")  # noqa: TRY002 - escapes `_step`
        if self.fail:
            raise RuntimeError("the step broke")
        self.steps += 1
        rows, seq = input_ids.shape
        # Write the token into the cache and read it back out of the cell the
        # batch names, so a mis-paged write becomes a wrong token.
        value = input_ids.float()[..., None, None].expand(rows, seq, 1, 8)
        batch = cache.batch(positions)
        cache[0].write(batch, torch.zeros_like(value), value.clone())
        last = cache[0].v.reshape(-1, 1, 8)[batch.cells[:, -1], 0, 0]
        logits = torch.zeros(rows, seq, VOCAB)
        logits[:, -1, :] = torch.nn.functional.one_hot(
            (last.long() + 1) % VOCAB, VOCAB
        ).float()
        return logits


def _scheduler(
    max_batch_size=4, max_seq_len=64, model=None, prefill_chunk=2048, kv_tokens=None
):
    return Scheduler(
        model or _StepModel(),
        torch.device("cpu"),
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        cuda_graph=False,
        compile=False,
        prefill_chunk=prefill_chunk,
        kv_tokens=kv_tokens,
    )


def _request(start, tokens=4):
    return Request(
        prompt=torch.tensor([[start]]),
        params=SamplingParams(max_new_tokens=tokens, temperature=0.0),
        stop_ids=frozenset(),
    )


def _expected(start, tokens=4):
    return [(start + i + 1) % VOCAB for i in range(tokens)]


def _drain(scheduler, requests):
    """Run every request at once and return each one's tokens."""
    out: dict[int, list[int]] = {}

    def drive(index, request):
        scheduler.submit(request)
        out[index] = list(request.stream())

    threads = [
        threading.Thread(target=drive, args=(i, r)) for i, r in enumerate(requests)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [out[i] for i in range(len(requests))]


def test_a_lone_request_generates_its_own_sequence():
    scheduler = _scheduler()
    try:
        assert _drain(scheduler, [_request(5)]) == [_expected(5)]
    finally:
        scheduler.close()


def test_batched_requests_do_not_read_each_others_pages():
    """The bug this guards: a row indexing the pool by anything but its own
    pages, which a uniform batch would hide."""
    scheduler = _scheduler()
    starts = [3, 17, 41, 58]
    try:
        got = _drain(scheduler, [_request(s) for s in starts])
    finally:
        scheduler.close()
    assert got == [_expected(s) for s in starts]


def test_more_requests_than_rows_still_all_run():
    scheduler = _scheduler(max_batch_size=2)
    starts = [1, 2, 3, 4, 5, 6]
    try:
        got = _drain(scheduler, [_request(s) for s in starts])
    finally:
        scheduler.close()
    assert got == [_expected(s) for s in starts]


def test_more_requests_than_pages_still_all_run():
    """Rows are not the only thing a request waits for now. Two pages of
    key/value serve two sequences at a time, and the rest queue behind them —
    on a scheduler with four rows standing idle, so it is the pages doing it.
    """
    scheduler = _scheduler(max_batch_size=4, kv_tokens=2 * PAGE_SIZE)
    assert scheduler.pool.pages == 2
    starts = [1, 2, 3, 4, 5, 6]
    try:
        got = _drain(scheduler, [_request(s) for s in starts])
    finally:
        scheduler.close()
    assert got == [_expected(s) for s in starts]


def test_a_request_larger_than_the_whole_pool_is_refused():
    """Refused at submit rather than queued: no sequence retiring will ever
    make room for it, so waiting would be waiting forever."""
    scheduler = _scheduler(max_seq_len=4096, kv_tokens=PAGE_SIZE)
    try:
        with pytest.raises(RequestError, match="pages of key/value"):
            scheduler.submit(_request(5, tokens=2 * PAGE_SIZE))
    finally:
        scheduler.close()


def test_a_request_takes_only_the_pages_it_asked_for():
    """The whole of what paging buys. A request for five tokens holds one page,
    where a slot pool gave it the full context every other request might have
    needed."""
    scheduler = _scheduler(max_seq_len=4096, kv_tokens=64 * PAGE_SIZE)
    request = _request(5, tokens=4)
    try:
        scheduler.submit(request)
        stream = request.stream()
        next(stream)
        assert scheduler.pool.free_pages == 63
        stream.close()
    finally:
        scheduler.close()


def test_a_finished_request_frees_its_row_and_pages():
    scheduler = _scheduler()
    try:
        _drain(scheduler, [_request(5), _request(9)])
        assert scheduler.running == {}
        assert scheduler.pool.free_rows == 4
        assert scheduler.pool.free_pages == scheduler.pool.pages
    finally:
        scheduler.close()


def test_a_stop_token_ends_the_stream_after_yielding_it():
    scheduler = _scheduler()
    request = Request(
        prompt=torch.tensor([[5]]),
        params=SamplingParams(max_new_tokens=8, temperature=0.0),
        stop_ids=frozenset({8}),
    )
    try:
        scheduler.submit(request)
        assert list(request.stream()) == [6, 7, 8]
    finally:
        scheduler.close()


def test_abandoning_a_stream_frees_the_row():
    """A disconnected client must stop occupying a row, not decode to its
    token limit with nowhere to put the tokens."""
    scheduler = _scheduler()
    request = _request(5, tokens=32)
    try:
        scheduler.submit(request)
        stream = request.stream()
        next(stream)
        stream.close()
        scheduler.close()
        assert scheduler.running == {}
        assert scheduler.pool.free_rows == 4
        assert scheduler.pool.free_pages == scheduler.pool.pages
    finally:
        scheduler.close()


def test_a_request_longer_than_the_pool_is_rejected():
    scheduler = _scheduler(max_seq_len=16)
    try:
        request = Request(
            prompt=torch.zeros(1, 12, dtype=torch.long),
            params=SamplingParams(max_new_tokens=8),
            stop_ids=frozenset(),
        )
        try:
            scheduler.submit(request)
        except RequestError as exc:
            assert "16-token context" in str(exc)
        else:
            raise AssertionError("an oversized request was accepted")
    finally:
        scheduler.close()


def test_a_failed_step_reaches_the_requests_it_broke():
    scheduler = _scheduler()
    request = _request(5, tokens=8)
    scheduler.submit(request)
    stream = request.stream()
    next(stream)
    scheduler.model.fail = True  # break the next step
    try:
        list(stream)
    except Exception as exc:
        assert not isinstance(exc, StopIteration)
    else:
        raise AssertionError("a broken step ended the stream as if it succeeded")
    finally:
        scheduler.close()


def test_buckets_are_powers_of_two_up_to_the_limit():
    assert buckets(8) == [1, 2, 4, 8]
    assert buckets(1) == [1]
    assert buckets(6) == [1, 2, 4, 6]  # the limit itself, not the next power
    assert buckets(1024, smallest=256) == [256, 512, 1024]


def test_a_dead_loop_fails_its_requests_instead_of_hanging():
    """The bug this guards: the loop is the only thing that puts tokens on a
    request's queue, so a loop that dies without saying so leaves every caller
    blocked on a `get` that never returns — a server that stops answering
    rather than one that fails."""
    scheduler = _scheduler()
    request = _request(5, tokens=32)
    scheduler.submit(request)
    stream = request.stream()
    next(stream)
    # Not an Exception: `_step` handles those per-request. This kills the loop.
    scheduler.model.crash = True

    with pytest.raises(BaseException, match="the loop died"):
        list(stream)

    # And a request arriving after the loop is gone is refused, not queued.
    with pytest.raises(RequestError, match="stopped"):
        scheduler.submit(_request(9))


def test_a_seeded_request_gets_its_own_generator():
    """`sample_batch` keys a seeded row to its generator; without one built
    here the seed would be accepted and silently ignored."""
    scheduler = _scheduler()
    try:
        request = Request(
            prompt=torch.tensor([[5]]),
            params=SamplingParams(max_new_tokens=1, temperature=1.0, seed=7),
            stop_ids=frozenset(),
        )
        scheduler.submit(request)
        assert request.generator is not None
        list(request.stream())
    finally:
        scheduler.close()


class _SumCache(StateCache):
    """A stand-in for recurrent state: one running total per slot.

    `KVCache` cannot show what a chunked prefill risks — its writes are indexed
    by position, so a stray step lands where the next chunk writes anyway. This
    is the other kind of state: a total that only moves forward, where a step
    the sequence did not ask for cannot be undone by writing over it.
    """

    def __init__(self, rows: int) -> None:
        self.total = torch.zeros(rows)

    def buffers(self) -> list:
        return [self.total]

    def view(self, start: int, stop: int) -> "_SumCache":
        cache = _SumCache.__new__(_SumCache)
        cache.total = self.total[start:stop]
        cache.primed = self.primed
        return cache


class _RecurrentModel:
    """Emits ``(state + 1) % VOCAB`` from state no position can repair.

    Every call advances the state, by its own tokens and by one for having run
    at all — so a step taken on a row in the middle of its prefill shows up in
    that row's next token.
    """

    def __init__(self) -> None:
        self.sampler = Sampler()
        self.eos_token_id = None

    def make_cache(self, max_batch_size, max_seq_len, pages=None):
        return CachePool(
            [_SumCache(max_batch_size)], max_batch_size, max_seq_len, pages
        )

    def __call__(self, input_ids, positions, cache):
        total = cache[0].total
        total += input_ids.float().sum(-1) + 1
        batch, _ = input_ids.shape
        logits = torch.zeros(batch, 1, VOCAB)
        return logits.scatter(2, ((total.long() + 1) % VOCAB)[:, None, None], 1.0)


def _prompt(values, tokens=4):
    return Request(
        prompt=torch.tensor([values]),
        params=SamplingParams(max_new_tokens=tokens, temperature=0.0),
        stop_ids=frozenset(),
    )


def test_a_chunked_prefill_generates_what_one_pass_generates():
    """The chunk is a scheduling decision, not an arithmetic one: the prompt
    attends over everything written before it either way."""
    prompt = [(i * 7) % VOCAB for i in range(10)]
    whole = _scheduler(prefill_chunk=64)
    try:
        expected = _drain(whole, [_prompt(prompt)])
    finally:
        whole.close()
    for chunk in (1, 3, 4, 9):
        scheduler = _scheduler(prefill_chunk=chunk)
        try:
            assert _drain(scheduler, [_prompt(prompt)]) == expected
        finally:
            scheduler.close()


class _Recording(_StepModel):
    """`_StepModel`, keeping the width of every forward it is asked for: a
    prompt chunk is as wide as the chunk, a decode step is one token."""

    def __init__(self) -> None:
        super().__init__()
        self.widths: list[int] = []
        self.positions: list[torch.Tensor] = []

    def __call__(self, input_ids, positions, cache):
        self.widths.append(input_ids.shape[1])
        self.positions.append(positions.clone())
        return super().__call__(input_ids, positions=positions, cache=cache)


def test_a_long_prefill_lets_the_batch_decode_between_its_chunks():
    """The point of the chunk. Unchunked, every running sequence waits out the
    whole prompt, and sees the wait as one gap between two of its tokens."""
    model = _Recording()
    scheduler = _scheduler(model=model, prefill_chunk=2)
    running = _request(5, tokens=48)
    try:
        scheduler.submit(running)
        stream = running.stream()
        next(stream)  # the batch is decoding before the long prompt arrives
        _drain(scheduler, [_prompt([(i * 5) % VOCAB for i in range(12)], tokens=2)])
        stream.close()
    finally:
        scheduler.close()
    chunks = [i for i, width in enumerate(model.widths) if width > 1]
    assert len(chunks) == 6  # 12 prompt tokens, 2 at a time
    assert any(model.widths[i] == 1 for i in range(chunks[0], chunks[-1]))


def test_a_prefill_between_chunks_keeps_the_state_it_has_built():
    """A step runs whole buckets, so a bucket wide enough to reach the slot
    being prefilled steps that slot too. What it advances has to be put back.

    Three neighbours, because that is what it takes: a decode step rounds its
    batch up to a bucket, and only the round-up reaches a slot no sequence is
    running in yet.
    """
    prompt = [(i * 3) % VOCAB for i in range(8)]
    alone = _scheduler(model=_RecurrentModel(), prefill_chunk=2)
    try:
        expected = _drain(alone, [_prompt(prompt)])
    finally:
        alone.close()

    scheduler = _scheduler(model=_RecurrentModel(), prefill_chunk=2)
    neighbours = [_prompt([i], tokens=48) for i in range(3)]
    streams = []
    try:
        for neighbour in neighbours:
            scheduler.submit(neighbour)
            stream = neighbour.stream()
            next(stream)  # admitted, and decoding
            streams.append(stream)
        assert scheduler.pool.free_rows == 1
        assert _drain(scheduler, [_prompt(prompt)]) == expected
        for stream in streams:
            stream.close()
    finally:
        scheduler.close()


def test_a_stepped_prefill_row_writes_where_its_next_chunk_writes():
    """The other half of the guard. A step writes a token into the row being
    prefilled, and a positional cache keeps whatever it is given: pointed at
    the start of the sequence it would overwrite the prompt, so it is pointed
    at the position the next chunk covers instead.
    """
    model = _Recording()
    scheduler = _scheduler(model=model, prefill_chunk=2)
    neighbours = [_request(i + 1, tokens=48) for i in range(3)]
    streams = []
    try:
        for neighbour in neighbours:
            scheduler.submit(neighbour)
            stream = neighbour.stream()
            next(stream)
            streams.append(stream)
        _drain(scheduler, [_prompt([(i * 3) % VOCAB for i in range(8)], tokens=2)])
        for stream in streams:
            stream.close()
    finally:
        scheduler.close()

    prefilled = 0
    steps = 0
    for width, positions in zip(model.widths, model.positions, strict=True):
        if width > 1:
            prefilled += width
        elif 0 < prefilled < 8:  # a step taken mid-prompt, slot 3 not yet live
            assert int(positions[3]) == prefilled
            steps += 1
    assert steps  # the bucket did reach the prefilling row


def test_a_prefill_chunk_of_zero_is_rejected():
    with pytest.raises(ValueError, match="prefill_chunk"):
        _scheduler(prefill_chunk=0)
